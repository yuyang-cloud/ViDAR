# ray normalization.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.models import HEADS
from mmcv.cnn import Linear, bias_init_with_prob
import math
import time
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from ...utils import e2e_predictor_utils
from ...modules.SAN_SAW import SAN, SAW
from einops import rearrange, repeat

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def fourier_embedding(input, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal input embeddings.

    :param input: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(input, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=input.device)
        args = input[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding

class Fourier_Embed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return fourier_embedding(t, self.dim)
    
@HEADS.register_module()
class LatentRenderingV2(BaseModule):
    """Ray marching adaptor for fine-tuning weights pre-trained by DriveGPT."""

    def __init__(self,
                 occ_flow='occ',
                 embed_dims=256,
                 num_pred_fcs=2,
                 pred_height=1,
                 num_cls=8,
                 grid_num=128,
                 grid_step=0.5,
                 reduction=16,
                 act='exp',
                 occ_render=False,
                 sem_render=False,
                 sem_norm=False,
                 sem_gt_train=True,
                 san_saw=False,
                 ego_motion_ln=False,
                 obj_motion_ln=False,

                 viz_response=False,
                 init_cfg=None):

        super().__init__(init_cfg)

        self.occ_flow = occ_flow
        self.embed_dims = embed_dims
        self.num_pred_fcs = num_pred_fcs    # 0
        self.grid_num = grid_num            # 256
        self.grid_step = grid_step          # 0.5
        self.viz_response = viz_response
        self.occ_render = occ_render
        self.sem_render = sem_render
        self.sem_norm = sem_norm
        self.sem_gt_train = sem_gt_train
        self.san_saw = san_saw
        self.ego_motion_ln = ego_motion_ln
        self.obj_motion_ln = obj_motion_ln

        # Activation function should be:
        #  'exp' or 'sigmoid'
        self.act = act

        if self.occ_render and self.occ_flow=='occ':
            # build up prob layer.
            unsup_raymarching_branch = []
            for _ in range(self.num_pred_fcs):
                unsup_raymarching_branch.append(Linear(self.embed_dims, self.embed_dims))
                unsup_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                unsup_raymarching_branch.append(nn.ReLU(inplace=True))
            unsup_raymarching_branch.append(Linear(self.embed_dims, pred_height))   # C -> D=16
            self.unsup_raymarching_head = nn.Sequential(*unsup_raymarching_branch)
            self.pred_height = pred_height  # 16

            # # LoRA layers.
            # self.lora_a = Linear(self.embed_dims, self.embed_dims // reduction)
            # self.lora_b = Linear(self.embed_dims // reduction, self.embed_dims)
        
        if self.sem_render and self.occ_flow=='occ':
            # build up prob layer.
            sem_raymarching_branch = []
            for _ in range(self.num_pred_fcs+1):
                sem_raymarching_branch.append(Linear(self.embed_dims, self.embed_dims))
                sem_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                sem_raymarching_branch.append(nn.ReLU(inplace=True))
            sem_raymarching_branch.append(Linear(self.embed_dims, num_cls))   # C -> cls
            self.sem_raymarching_branch = nn.Sequential(*sem_raymarching_branch)
            self.num_cls = num_cls-1  # cls
            self.sem_hidden_dim = self.embed_dims // self.num_cls

            self.sem_lora_a = Linear(self.embed_dims, self.num_cls * self.sem_hidden_dim)
            self.sem_group_conv = nn.Conv2d(self.num_cls*self.sem_hidden_dim, self.num_cls*self.sem_hidden_dim, kernel_size=1, groups=self.num_cls)
            self.sem_lora_b = nn.Conv2d(self.num_cls*self.sem_hidden_dim, self.embed_dims, kernel_size=1)
        elif self.sem_norm and self.occ_flow=='occ':
            # build up prob layer.
            sem_raymarching_branch = []
            for _ in range(self.num_pred_fcs+1):
                sem_raymarching_branch.append(Linear(self.embed_dims, self.embed_dims))
                sem_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                sem_raymarching_branch.append(nn.ReLU(inplace=True))
            sem_raymarching_branch.append(Linear(self.embed_dims, num_cls))   # C -> cls
            self.sem_raymarching_branch = nn.Sequential(*sem_raymarching_branch)
            self.num_cls = num_cls

            self.sem_param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 128
            self.sem_mlp_shared = nn.Sequential(
                nn.Conv2d(self.num_cls, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.sem_mlp_gamma = nn.Conv2d(nhidden, self.embed_dims, kernel_size=3, padding=1)
            self.sem_mlp_beta = nn.Conv2d(nhidden, self.embed_dims, kernel_size=3, padding=1)
            # self.reset_parameters()
            nn.init.zeros_(self.sem_mlp_gamma.weight)
            nn.init.zeros_(self.sem_mlp_beta.weight)
            nn.init.ones_(self.sem_mlp_gamma.bias)
            nn.init.zeros_(self.sem_mlp_beta.bias)
        
        if self.san_saw and self.occ_flow == 'occ':
            self.num_cls = num_cls
            self.classifier = nn.Conv2d(self.embed_dims, num_cls, kernel_size=1, stride=1, bias=True)

            self.selecte_classes = [1,2,3,4,5,6,7,8]
            self.san_stage = SAN(inplanes=self.embed_dims, selected_classes=self.selecte_classes)
            # self.saw_stage = SAW(dim=self.embed_dims, selected_classes=self.selecte_classes, relax_denom=2.0, classifier=self.classifier)
        
        if self.ego_motion_ln:
            self.ego_param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 256
            self.ego_mlp_shared = nn.Sequential(
                nn.Linear(144, nhidden),
                nn.ReLU(),
            )
            self.ego_mlp_gamma = nn.Linear(nhidden, nhidden)
            self.ego_mlp_beta = nn.Linear(nhidden, nhidden)
            # self.reset_parameters()
            nn.init.zeros_(self.ego_mlp_gamma.weight)
            nn.init.zeros_(self.ego_mlp_beta.weight)
            nn.init.ones_(self.ego_mlp_gamma.bias)
            nn.init.zeros_(self.ego_mlp_beta.bias)
        
        if self.obj_motion_ln and self.occ_flow=='flow':
            self.fourier_embed = Fourier_Embed(32)
            
            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            self.pred_height = pred_height  # 16
            self.mlp_shared = nn.Sequential(
                nn.Linear(3*32, 64),
                nn.ReLU(),
                nn.Linear(64, self.embed_dims // self.pred_height),
                nn.ReLU(),
            )
            self.mlp_gamma = nn.Linear(self.embed_dims // self.pred_height, self.embed_dims // self.pred_height)
            self.mlp_beta = nn.Linear(self.embed_dims // self.pred_height, self.embed_dims // self.pred_height)
            self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.ones_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.bias)


    def forward_occ_render(self,
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
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(occ_pred[0].max(0)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()

        # 2. get query positions.
        occ_grids = e2e_predictor_utils.get_bev_grids(bev_h, bev_w, bs=bs, offset=0.5)  # bs, h*w, 2 每个bev grid的坐标 左上角为(0,0) 右下角为(1,1)
        occ_grids = occ_grids - 0.5
        # occ_grids = torch.nan_to_num(
        #     occ_grids / torch.sqrt((occ_grids ** 2).sum(-1, keepdims=True)))    # bs, h*w, 2        bev_grid坐标归一化
        prev_step = self.grid_step / (min(bev_h, bev_w) // 2)                       # 0.5/100 = 0.005
        prev_step = torch.from_numpy(np.arange(0, self.grid_num) + 0.5).to(         # 256
            occ_grids.dtype).to(occ_grids.device) * prev_step
        occ_path_grids = occ_grids.view(bs, -1, 1, 2) * prev_step.view(1, 1, -1, 1)  # bs, h*w, num_grid, 2

        occ_path_grids = torch.cat([occ_path_grids, occ_grids.view(bs, bev_h * bev_w, 1, 2)], 2)    # bs, h*w, num_grid+1, 2  每个点的 同一个ray上的前边采样点+自己 坐标
        occ_path_grids = occ_path_grids * 2  # norm to [-1, 1] for F.upsample.
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
        occ_path_prob = occ_path_prob.view(bs, bev_w, bev_h, self.pred_height)                  # B,H,W,D   conditional_prob = cigma_{j=0~i}(1-p_j) * p_i
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(occ_path_prob[0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        # # 5. (Additional operations): find the ray features & distribute to each point.
        # # remove the current points.
        # occ_path_grids = occ_path_grids[..., :-1, :].contiguous()   # bs, h*w, num_grid, 2  每个点的 同一个ray上的前边采样点 坐标
        # # Add-1: get features of each point.
        # embed = self.lora_a(embed)  # bs, bev_h, bev_w, reduction_dim
        # embed = embed.permute(0, 3, 1, 2).contiguous()  # bs, embed_dim, bev_h, bev_w
        # embed = F.grid_sample(embed, occ_path_grids, align_corners=False)  # bs, embed_dim, h*w, num_grid
        # # Add-2: get prob of each point.
        # occ_path_boundary = torch.minimum(1 / torch.abs(occ_grids[..., 0:1]),
        #                                   1 / torch.abs(occ_grids[..., 1:2]))  # bs, -1
        # occ_path_valid_mask = (occ_path_length[..., :-1, :] < occ_path_boundary.view(bs, -1, 1, 1))
        # # bs, num_height, num_points, num_grids
        # occ_path_prob_grids = F.grid_sample(
        #     occ_path_prob.permute(0, 3, 1, 2).contiguous(),
        #     occ_path_grids, align_corners=False)    # bs, num_height, h*w, num_grids  每个点的 同一个ray上的前边采样点的 conditional_prob
        # occ_path_prob_grids = (occ_path_prob_grids *
        #                        occ_path_valid_mask.view(bs, 1, bev_h * bev_w, self.grid_num))
        # occ_path_prob_grids = occ_path_prob_grids / (occ_path_prob_grids.sum(-1, keepdims=True) + eps)
        # embed = (embed.view(bs, self.pred_height, -1, bev_h * bev_w, self.grid_num) *
        #          occ_path_prob_grids.view(bs, self.pred_height, 1, bev_h * bev_w, self.grid_num))
        # embed = embed.view(bs, -1, bev_h * bev_w, self.grid_num)  # bs, embed_dim, h*w, num_grids 每个点的 同一个ray上的前边采样点的conditional_prob*feat
        # embed = embed.sum(-1)  # bs, embed_dim, h*w  每个点的 同一个ray上前边采样点的sum(conditional_prob*feat)
        # # Add-3: lora back.
        # embed = embed.permute(0, 2, 1).contiguous()  # bs, num_point, embed_dim
        # embed = self.lora_b(embed)
        # embed = embed.view(bs, bev_h, bev_w, self.embed_dims)

        # 6. get final embedding.
        embed_shape = embed.shape # bs, h, w, C
        embed = (embed.view(bs, bev_h, bev_w, self.pred_height, -1) * # bev_feat * condional_prob = B,H,W,D,C//D * B,H,W,D,1 = B,H,W,D,C//D
                 occ_path_prob.view(bs, bev_h, bev_w, self.pred_height, 1))
        embed = embed.view(embed_shape)  # B,H,W,C
        return embed, occ_pred
    
    def forward_sem_render(self,
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
        sem_pred = self.sem_raymarching_branch(embed)  # bs, bev_h, bev_w, cls+1
        sem_pred_logits = torch.softmax(sem_pred, dim=-1)
        sem_pred_logits = sem_pred_logits.permute(0, 3, 1, 2)[:, 1:, ...].contiguous()  # bs, cls, bev_h, bev_w

        # 2. get query positions.
        occ_grids = e2e_predictor_utils.get_bev_grids(bev_h, bev_w, bs=bs, offset=0.5)  # bs, h*w, 2 每个bev grid的坐标 左上角为(0,0) 右下角为(1,1)
        occ_grids = occ_grids - 0.5
        occ_grids = torch.nan_to_num(
            occ_grids / torch.sqrt((occ_grids ** 2).sum(-1, keepdims=True)))    # bs, h*w, 2        bev_grid坐标归一化
        prev_step = self.grid_step / (min(bev_h, bev_w) // 2)                       # 0.5/100 = 0.005
        prev_step = torch.from_numpy(np.arange(0, self.grid_num) + 0.5).to(         # 256
            occ_grids.dtype).to(occ_grids.device) * prev_step
        occ_path_grids = 0.5 + occ_grids.view(bs, -1, 1, 2) * prev_step.view(1, 1, -1, 1)  # bs, h*w, num_grid, 2

        occ_path_grids = torch.cat([occ_path_grids, occ_grids.view(bs, bev_h * bev_w, 1, 2)], 2)    # bs, h*w, num_grid+1, 2  每个点的 同一个ray上的前边采样点+自己 坐标
        occ_path_grids = occ_path_grids * 2 - 1  # norm to [-1, 1] for F.upsample.
        occ_path_per_prob = F.grid_sample(sem_pred_logits, occ_path_grids, align_corners=False)  # bs, cls, h*w, num_grid+1 
        occ_path_per_prob = occ_path_per_prob.permute(0, 2, 3, 1)  # bs, h*w, num_grid+1, cls 每个点的 同一个ray上的前边采样点+自己 的occ （注意有的采样点超出 自己，采出来是0）

        # 3. get prob, and sum those all.
        #  ignore waypoints outside the current grid.
        occ_path_length = torch.sqrt((occ_path_grids ** 2).sum(-1, keepdims=True))  # bs, h*w, num_grids+1, 1
        occ_path_valid_mask = (occ_path_length < occ_path_length[..., -1:, :])      # bs, h*w, num_grids+1, 1  同一个ray上的前边采样点 超出 自己的 为False

        # 4. Ray-marching-accumulation.
        occ_path_prev_prob = torch.cumprod(1 - occ_path_per_prob * occ_path_valid_mask, dim=2)  # bs, h*w, num_grids+1, cls  同一个ray上的前边采样点 non-occ累积概率
        occ_path_prob = occ_path_prev_prob[..., -1, :] * occ_path_per_prob[..., -1, :]          # bs, h*w, cls   同一个ray上前边所有采样点non-occ累积概率 * 自己occ概率
        occ_path_prob = occ_path_prob.view(bs, bev_w, bev_h, self.num_cls).transpose(1,2)  # B,H,W,cls   conditional_prob = cigma_{j=0~i}(1-p_j) * p_i

        # # 5. (Additional operations): find the ray features & distribute to each point.
        # # remove the current points.
        # occ_path_grids = occ_path_grids[..., :-1, :].contiguous()   # bs, h*w, num_grid, 2  每个点的 同一个ray上的前边采样点 坐标
        # # Add-1: get features of each point.
        # embed = self.lora_a(embed)  # bs, bev_h, bev_w, reduction_dim
        # embed = embed.permute(0, 3, 1, 2).contiguous()  # bs, embed_dim, bev_h, bev_w
        # embed = F.grid_sample(embed, occ_path_grids, align_corners=False)  # bs, embed_dim, h*w, num_grid
        # # Add-2: get prob of each point.
        # occ_path_boundary = torch.minimum(1 / torch.abs(occ_grids[..., 0:1]),
        #                                   1 / torch.abs(occ_grids[..., 1:2]))  # bs, -1
        # occ_path_valid_mask = (occ_path_length[..., :-1, :] < occ_path_boundary.view(bs, -1, 1, 1))
        # # bs, num_height, num_points, num_grids
        # occ_path_prob_grids = F.grid_sample(
        #     occ_path_prob.permute(0, 3, 1, 2).contiguous(),
        #     occ_path_grids, align_corners=False)    # bs, num_height, h*w, num_grids  每个点的 同一个ray上的前边采样点的 conditional_prob
        # occ_path_prob_grids = (occ_path_prob_grids *
        #                        occ_path_valid_mask.view(bs, 1, bev_h * bev_w, self.grid_num))
        # occ_path_prob_grids = occ_path_prob_grids / (occ_path_prob_grids.sum(-1, keepdims=True) + eps)
        # embed = (embed.view(bs, self.pred_height, -1, bev_h * bev_w, self.grid_num) *
        #          occ_path_prob_grids.view(bs, self.pred_height, 1, bev_h * bev_w, self.grid_num))
        # embed = embed.view(bs, -1, bev_h * bev_w, self.grid_num)  # bs, embed_dim, h*w, num_grids 每个点的 同一个ray上的前边采样点的conditional_prob*feat
        # embed = embed.sum(-1)  # bs, embed_dim, h*w  每个点的 同一个ray上前边采样点的sum(conditional_prob*feat)
        # # Add-3: lora back.
        # embed = embed.permute(0, 2, 1).contiguous()  # bs, num_point, embed_dim
        # embed = self.lora_b(embed)
        # embed = embed.view(bs, bev_h, bev_w, self.embed_dims)

        # 6. get final embedding.
        embed =  self.sem_lora_a(embed)
        embed = (embed.view(bs, bev_h, bev_w, self.num_cls, self.sem_hidden_dim) * # bev_feat * condional_prob = B,H,W,Cls,C' * B,H,W,Cls,1 = B,H,W,Cls,C'
                 occ_path_prob.view(bs, bev_h, bev_w, self.num_cls, 1))
        embed = embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()  # B,Cls*C',H,W
        embed = self.sem_group_conv(embed)  # 按Cls分组Conv
        embed = self.sem_lora_b(embed).permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        return embed, sem_pred.permute(0, 3, 1, 2)
    
    def forward_sem_norm(self,
                        embed,
                        sem_label=None,
                        **kwargs):
        """Forward Function of RayNormalization.

        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
        """

        # 1. obtain unsupervised occupancy prediction.
        sem_pred = self.sem_raymarching_branch(embed)  # bs, bev_h, bev_w, cls+1
        if not self.sem_gt_train or sem_label is None:
            sem_label = torch.argmax(sem_pred.detach(), dim=-1)     # bs, bev_h, bev_w
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(sem_label[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        sem_code = F.one_hot(sem_label.long(), num_classes=self.num_cls).float().permute(0,3,1,2).contiguous()

        # 2. generate parameter-free normalized activations
        embed = self.sem_param_free_norm(embed)
        embed = embed.permute(0,3,1,2)
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(0)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()

        # 3. produce scaling and bias conditioned on semantic map
        actv = self.sem_mlp_shared(sem_code)
        gamma = self.sem_mlp_gamma(actv)
        beta = self.sem_mlp_beta(actv)

        # apply scale and bias
        embed = gamma * embed + beta
        embed = embed.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        return embed, sem_pred.permute(0, 3, 1, 2)

    def forward_san_saw(self,
                        embed,
                        **kwargs):
        """Forward Function of RayNormalization.

        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
        """
        embed = embed.permute(0,3,1,2)  # B,C,H,W
        embed_ori = embed

        # 1. classfier
        sem_pred = self.classifier(embed.detach())  # B,cls,H,W
        
        # 2. SAN
        embed = self.san_stage(embed, sem_pred) # B,C,H,W
        embed_san = embed

        # 3. SAW
        # saw_loss = self.saw_stage(embed)

        embed = embed.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        if self.training:
            out_dict = {
                'sem_pred': sem_pred,   # B,cls,H,W
                'embed_ori': embed_ori, # B,C,H,W
                'embed_san': embed_san, # B,C,H,W
                # 'saw_loss': torch.tensor([0]).to(embed),   # 1
            }
        else:
            out_dict = None
        return embed, out_dict
    
    def forward_ego_motion_ln(self,
                            embed,
                            future2history=None,
                            **kwargs):
        """Forward Function of RayNormalization.

        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
            future2history: bs, 4, 4
        """
        # 1. memory_ego_motion
        memory_ego_motion = future2history[:, :3, :].flatten(-2).float()   # B,12
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)

        # 2. generate parameter-free normalized activations
        embed = self.ego_param_free_norm(embed)

        # 3. produce scaling and bias conditioned on semantic map
        actv = self.ego_mlp_shared(memory_ego_motion)   # B,C
        gamma = self.ego_mlp_gamma(actv)
        beta = self.ego_mlp_beta(actv)

        # apply scale and bias
        embed = gamma * embed + beta
        return embed

    def forward_obj_motion_ln(self,
                            embed,
                            flow_3D=None,
                            occ_3D=None,
                            **kwargs):
        """Forward Function of RayNormalization.

        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
            flow_3D: bs, 3, H, W, D
            occ_3D: bs, H, W, D
        """
        # 1. rearrange
        flow_3D = flow_3D.permute(0,2,3,4,1)
        b, h, w, d, dims = flow_3D.shape
        flow_3D = flow_3D * (occ_3D > 0).float().unsqueeze(-1)
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(flow_3D[0].mean(2)[...,1].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        flow_3D = rearrange(flow_3D, "b h w d c -> (b h w d c)")

        # 2. fourier embed
        flow_3D = self.fourier_embed(flow_3D)
        flow_3D = rearrange(flow_3D, "(b h w d c) c2 -> b h w d (c c2)", b=b, h=h, w=w, d=d, c=dims, c2=self.fourier_embed.dim)

        # 3. generate parameter-free normalized activations
        embed = self.param_free_norm(embed)
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()

        # 4. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(flow_3D)   # B,H,W,D,C'
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        embed = gamma * embed.view(b, h, w, d, -1) + beta
        embed = embed.view(b, h, w, -1)
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        return embed
    
    def forward(self, embeds, bev_sem_gts=None, flow_3D=None, occ_3D=None, future2history=None):
        """Forward Function of Rendering.

        Args:
            embeds (Tensor): BEV feature embeddings.
                `(bs, F, bev_h, bev_w, embed_dims)`
        """
        bs, num_frames, bev_h, bev_w, embed_dim = embeds.shape

        # occ rendering
        if self.occ_render and self.occ_flow=='occ':
            occ_embeds = []
            for i in range(num_frames):
                occ_embed, bev_occ_pred = self.forward_occ_render(embeds[:, i, ...])
                occ_embeds.append(occ_embed)
            occ_embeds = torch.stack(occ_embeds, dim=1)

        if self.sem_render and self.occ_flow=='occ':
            sem_embeds = []
            for i in range(num_frames):
                sem_embed, bev_sem_pred = self.forward_sem_render(embeds[:, i, ...])
                sem_embeds.append(sem_embed)
            sem_embeds = torch.stack(sem_embeds, dim=1)

        if self.sem_norm and self.occ_flow=='occ':
            sem_embeds = []
            for i in range(num_frames):
                if bev_sem_gts is not None:
                    sem_embed, bev_sem_pred = self.forward_sem_norm(embeds[:, i, ...], bev_sem_gts[:, i, ...])
                else:
                    sem_embed, bev_sem_pred = self.forward_sem_norm(embeds[:, i, ...])
                sem_embeds.append(sem_embed)
            sem_embeds = torch.stack(sem_embeds, dim=1)

        if self.san_saw and self.occ_flow=='occ':
            embeds = embeds.flatten(0, 1)
            sem_embeds, san_saw_output = self.forward_san_saw(embeds)
            sem_embeds = sem_embeds.view(bs, num_frames, bev_h, bev_w, embed_dim)

        # if self.occ_render and self.sem_render and self.occ_flow=='occ':
        #     embeds = occ_embeds + sem_embeds
        # elif self.occ_render and self.occ_flow=='occ':
        #     embeds = occ_embeds
        # elif self.sem_render and self.occ_flow=='occ':
        #     embeds = sem_embeds
        # elif self.sem_norm and self.occ_flow=='occ':
        #     embeds = sem_embeds
        # elif self.san_saw and self.occ_flow=='occ':
        #     embeds = sem_embeds


        if self.ego_motion_ln and future2history is not None:
            ego_motion_embeds = []
            for i in range(num_frames):
                motion_embed = self.forward_ego_motion_ln(embeds[:, i, ...], future2history[:, i, ...])
                ego_motion_embeds.append(motion_embed)
            ego_motion_embeds = torch.stack(ego_motion_embeds, dim=1)

        if self.obj_motion_ln and flow_3D is not None:
            obj_motion_embeds = []
            for i in range(num_frames):
                motion_embed = self.forward_obj_motion_ln(embeds[:, i, ...], flow_3D[:, i, ...], occ_3D[:, i, ...])
                obj_motion_embeds.append(motion_embed)
            obj_motion_embeds = torch.stack(obj_motion_embeds, dim=1)


        if self.sem_norm and self.occ_flow=='occ' and self.ego_motion_ln and future2history is not None:
            embeds = sem_embeds + ego_motion_embeds
        elif self.sem_norm and self.occ_flow=='occ':
            embeds = sem_embeds
        elif self.ego_motion_ln and future2history is not None:
            embeds = ego_motion_embeds

        # if (self.ego_motion_ln and future2history is not None) and (self.obj_motion_ln and flow_3D is not None):
        #     embeds = ego_motion_embeds + obj_motion_embeds
        # elif self.ego_motion_ln and future2history is not None:
        #     embeds = ego_motion_embeds
        # elif self.obj_motion_ln and flow_3D is not None: 
        #     embeds = obj_motion_embeds

        embeds = embeds.transpose(2, 3).contiguous() # NOTE: query first_H, then_W
        out_dict = {
            'bev_embed': embeds.view(bs, num_frames, -1, embed_dim),    # B,F,HW,C
            'bev_occ_pred': bev_occ_pred if self.occ_render and self.occ_flow=='occ' and self.training else None,  # B,D,H,W       last_frame
            'bev_sem_pred': bev_sem_pred if (self.sem_render or self.sem_norm) and self.occ_flow=='occ' and self.training else None,  # B,cls,H,W     last_frame
            'san_saw_output': san_saw_output if self.san_saw and self.occ_flow=='occ' and self.training else None,
        }
        if self.occ_render or self.sem_render or (self.obj_motion_ln and flow_3D is not None):
            # Save GPU memory
            torch.cuda.empty_cache()
        return out_dict
