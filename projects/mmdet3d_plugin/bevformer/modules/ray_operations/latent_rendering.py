# ray normalization.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn import Linear, bias_init_with_prob
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from ...utils import e2e_predictor_utils


@ATTENTION.register_module()
class LatentRendering(BaseModule):
    """Ray marching adaptor for fine-tuning weights pre-trained by DriveGPT."""

    def __init__(self,
                 embed_dims=256,
                 num_pred_fcs=2,
                 pred_height=1,
                 grid_num=128,
                 grid_step=0.5,
                 reduction=16,
                 act='exp',

                 viz_response=False,
                 init_cfg=None):

        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_pred_fcs = num_pred_fcs    # 0
        self.grid_num = grid_num            # 256
        self.grid_step = grid_step          # 0.5
        self.viz_response = viz_response

        # Activation function should be:
        #  'exp' or 'sigmoid'
        self.act = act

        # build up prob layer.
        unsup_raymarching_branch = []
        for _ in range(self.num_pred_fcs):
            unsup_raymarching_branch.append(Linear(self.embed_dims, self.embed_dims))
            unsup_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
            unsup_raymarching_branch.append(nn.ReLU(inplace=True))
        unsup_raymarching_branch.append(Linear(self.embed_dims, pred_height))   # C -> D=16
        self.unsup_raymarching_head = nn.Sequential(*unsup_raymarching_branch)
        self.pred_height = pred_height  # 16

        # LoRA layers.
        self.lora_a = Linear(self.embed_dims, self.embed_dims // reduction)
        self.lora_b = Linear(self.embed_dims // reduction, self.embed_dims)

    def forward(self,
                embed,
                eps=1e-3,
                **kwargs):
        """Forward Function of RayNormalization.

        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
            pos (Tensor): positions of each query point in feature embedding.
                `(bs, bev_h, bev_w, 2)`
        """
        bs, bev_h, bev_w, embed_dim = embed.shape

        # 1. obtain unsupervised occupancy prediction.
        occ_pred = self.unsup_raymarching_head(embed)  # bs, bev_h, bev_w, num_height
        occ_pred = occ_pred.permute(0, 3, 1, 2).contiguous()  # bs, num_height, bev_h, bev_w

        # 2. get query positions.
        occ_grids = e2e_predictor_utils.get_bev_grids(bev_h, bev_w, bs=bs, offset=0.5)  # bs, h*w, 2 每个bev grid的坐标 左上角为(0,0) 右下角为(1,1)
        occ_grids_r = occ_grids - 0.5
        occ_grid_r_norm = torch.nan_to_num(
            occ_grids_r / torch.sqrt((occ_grids_r ** 2).sum(-1, keepdims=True)))    # bs, h*w, 2        bev_grid坐标归一化
        prev_step = self.grid_step / (min(bev_h, bev_w) // 2)                       # 0.5/100 = 0.005
        prev_step = torch.from_numpy(np.arange(0, self.grid_num) + 0.5).to(         # 256
            occ_grid_r_norm.dtype).to(occ_grid_r_norm.device) * prev_step
        occ_prev_grids = 0.5 + occ_grid_r_norm.view(bs, -1, 1, 2) * prev_step.view(1, 1, -1, 1)  # bs, h*w, num_grid, 2

        occ_path_grids = torch.cat([occ_prev_grids, occ_grids.view(bs, bev_h * bev_w, 1, 2)], 2)    # bs, h*w, num_grid+1, 2  每个点的 同一个ray上的前边采样点+自己 坐标
        occ_path_grids = occ_path_grids * 2 - 1  # norm to [-1, 1] for F.upsample.
        occ_path_per_prob = F.grid_sample(occ_pred, occ_path_grids, align_corners=False)  # bs, num_height, h*w, num_grid+1 
        occ_path_per_prob = occ_path_per_prob.permute(0, 2, 3, 1)  # bs, h*w, num_grid+1, num_height 每个点的 同一个ray上的前边采样点+自己 的occ （注意有的采样点超出 自己，采出来是0）

        # 3. get prob, and sum those all.
        #  ignore waypoints outside the current grid.
        occ_path_length = torch.sqrt((occ_path_grids ** 2).sum(-1, keepdims=True))  # bs, h*w, num_grids+1, 1
        occ_path_valid_mask = (occ_path_length < occ_path_length[..., -1:, :])      # bs, h*w, num_grids+1, 1  同一个ray上的前边采样点 超出 自己的 为False
        #  activate the prob.
        if self.act == 'exp':
            occ_path_per_prob = F.relu(occ_path_per_prob, inplace=True)
            occ_path_per_prob = 1 - torch.exp(-occ_path_per_prob)  # inside prob
        elif self.act == 'sigmoid':
            occ_path_per_prob = torch.sigmoid(occ_path_per_prob)    # bs, h*w, num_grids+1, num_height
        else:
            raise NotImplementedError('Only support exp or sigmoid activation_fn for now.')

        # 4. Ray-marching-accumulation.
        occ_path_prev_prob = torch.cumprod(1 - occ_path_per_prob * occ_path_valid_mask, dim=2)  # bs, h*w, num_grids+1, num_height  同一个ray上的前边采样点 non-occ累积概率
        occ_path_prob = occ_path_prev_prob[..., -1, :] * occ_path_per_prob[..., -1, :]          # bs, h*w, num_height   同一个ray上前边所有采样点non-occ累积概率 * 自己occ概率
        occ_path_prob = occ_path_prob.view(bs, bev_h, bev_w, self.pred_height)  # B,H,W,D   conditional_prob = cigma_{j=0~i}(1-p_j) * p_i

        # 5. (Additional operations): find the ray features & distribute to each point.
        # remove the current points.
        occ_path_grids = occ_path_grids[..., :-1, :].contiguous()   # bs, h*w, num_grid, 2  每个点的 同一个ray上的前边采样点 坐标
        # Add-1: get features of each point.
        embed_after_lora_a = self.lora_a(embed)  # bs, bev_h, bev_w, reduction_dim
        embed_after_lora_a = embed_after_lora_a.permute(0, 3, 1, 2).contiguous()  # bs, embed_dim, bev_h, bev_w
        occ_path_embed_grids = F.grid_sample(embed_after_lora_a, occ_path_grids, align_corners=False)  # bs, embed_dim, h*w, num_grid
        # Add-2: get prob of each point.
        occ_path_boundary = torch.minimum(1 / torch.abs(occ_grid_r_norm[..., 0:1]),
                                          1 / torch.abs(occ_grid_r_norm[..., 1:2]))  # bs, -1
        occ_path_valid_mask = (occ_path_length[..., :-1, :] < occ_path_boundary.view(bs, -1, 1, 1))
        # bs, num_height, num_points, num_grids
        occ_path_prob_grids = F.grid_sample(
            occ_path_prob.permute(0, 3, 1, 2).contiguous(),
            occ_path_grids, align_corners=False)    # bs, num_height, h*w, num_grids  每个点的 同一个ray上的前边采样点的 conditional_prob
        occ_path_prob_grids = (occ_path_prob_grids *
                               occ_path_valid_mask.view(bs, 1, bev_h * bev_w, self.grid_num))
        occ_path_prob_grids = occ_path_prob_grids / (occ_path_prob_grids.sum(-1, keepdims=True) + eps)
        occ_path_embed = (occ_path_embed_grids.view(bs, self.pred_height, -1, bev_h * bev_w, self.grid_num) *
                          occ_path_prob_grids.view(bs, self.pred_height, 1, bev_h * bev_w, self.grid_num))
        occ_path_embed = occ_path_embed.view(bs, -1, bev_h * bev_w, self.grid_num)  # bs, embed_dim, h*w, num_grids 每个点的 同一个ray上的前边采样点的conditional_prob*feat
        occ_path_embed = occ_path_embed.sum(-1)  # bs, embed_dim, h*w  每个点的 同一个ray上前边采样点的sum(conditional_prob*feat)
        # Add-3: lora back.
        occ_path_embed = occ_path_embed.permute(0, 2, 1).contiguous()  # bs, num_point, embed_dim
        occ_path_embed = self.lora_b(occ_path_embed)
        occ_path_embed = occ_path_embed.view(bs, bev_h, bev_w, self.embed_dims)

        # 6. get final embedding.
        occ_path_embed_shape = occ_path_embed.shape # bs, h, w, C
        occ_path_embed = (occ_path_embed.view(bs, bev_h, bev_w, self.pred_height, -1) * # bev_feat * condional_prob = B,H,W,D,C//D * B,H,W,D,1 = B,H,W,D,C//D
                          occ_path_prob.view(bs, bev_h, bev_w, self.pred_height, 1))
        occ_path_embed = occ_path_embed.view(occ_path_embed_shape)  # B,H,W,C
        return occ_path_embed
