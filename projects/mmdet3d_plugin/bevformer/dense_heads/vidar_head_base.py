#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
<Base class> of ViDAR future prediction head:
    * Future Decoder
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import xavier_init, constant_init
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from torch.nn.init import normal_
from ..utils import e2e_predictor_utils
from mmdet3d.models import builder
from mmdet3d.models.losses import chamfer_distance
from einops import rearrange, repeat
from ..modules.ray_operations.latent_rendering_v2 import Fourier_Embed as Fourier_Embed_v1
import math

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
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * self.embed_dim, 256),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            xavier_init(self.mlp, distribution='uniform', bias=0.)
        except:
            pass

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, c = t.shape
        # fourier embed
        t = rearrange(t, "b c -> (b c)")
        t = fourier_embedding(t, self.embed_dim)
        t = rearrange(t, "(b c) c2 -> b (c c2)", b=b, c=c, c2=self.embed_dim)
        # mlp
        t = self.mlp(t)
        return t

@HEADS.register_module()
class ViDARHeadTemplate(BaseModule):
    """Head of ViDAR: Visual Point Cloud Forecasting.
    """

    def __init__(self,
                 *args,
                 num_classes,
                 # Architecture.
                 prev_render_neck=None,
                 transformer=None,
                 num_pred_fcs=2,
                 num_pred_height=1,

                 # Memory Queue configurations.
                 memory_queue_len=1,
                 turn_on_flow=False,
                 sem_norm=False,
                 obj_motion_norm=False,

                 # Embedding configuration.
                 use_can_bus=False,
                 can_bus_norm=True,
                 can_bus_dims=(0, 1, 2, 17),
                 use_plan_traj=True,
                 condition_ca_add='add',
                 use_command=False,
                 use_vel_steering=False,
                 use_vel=False,
                 use_steering=False,

                 # target BEV configurations.
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,

                 # loss functions.
                 loss_weight=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # evaluation configuration.
                 eval_within_grid=False,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        # memory queue
        self.memory_queue_len = memory_queue_len
        self.turn_on_flow = turn_on_flow
        self.sem_norm = sem_norm
        self.obj_motion_norm = obj_motion_norm
        # fourier_embed
        self.fourier_nhidden = 64
        # Embedding configurations.
        self.can_bus_norm = can_bus_norm
        self.use_can_bus = use_can_bus
        self.can_bus_dims = can_bus_dims  # (delta_x, delta_y, delta_z, delta_yaw)
        if self.use_can_bus:
            self.fourier_embed_canbus = Fourier_Embed(len(self.can_bus_dims), self.fourier_nhidden)
        # vel_steering
        self.use_vel_steering = use_vel_steering
        self.vel_steering_dims = 4        # (vx, vy, v_yaw, steering)
        if self.use_vel_steering:
            self.fourier_embed_velsteering = Fourier_Embed(self.vel_steering_dims, self.fourier_nhidden)
        # vel
        self.use_vel = use_vel
        self.vel_dims = 3        # (vx, vy, v_yaw)
        if self.use_vel:
            self.fourier_embed_vel = Fourier_Embed(self.vel_dims, self.fourier_nhidden)
        # steering
        self.use_steering = use_steering
        self.steering_dims = 1        # (vx, vy, v_yaw, steering)
        if self.use_steering:
            self.fourier_embed_steering = Fourier_Embed(self.steering_dims, self.fourier_nhidden)
        # command
        self.use_command = use_command
        self.command_dims = 1             # command
        if self.use_command:
            self.fourier_embed_command = Fourier_Embed(self.command_dims, self.fourier_nhidden)
        # plan_traj
        self.use_plan_traj = use_plan_traj
        self.plan_traj_dims = 2
        if self.use_plan_traj:
            self.fourier_embed_plantraj = Fourier_Embed(self.plan_traj_dims, self.fourier_nhidden)
        # action_condition
        self.action_condition_dims = 256 * use_can_bus + 256 * use_vel_steering + 256 * use_vel + 256 * use_steering + 256 * use_command + 256 * use_plan_traj
        self.condition_ca_add = condition_ca_add

        # Network configurations.
        self.num_pred_fcs = num_pred_fcs
        # How many bins predicted at the height dimensions.
        # By default, 1 for BEV prediction.
        self.num_pred_height = num_pred_height

        # build prev_render_neck
        if prev_render_neck is not None:
            self.prev_render_neck = builder.build_head(prev_render_neck)
        else:
            self.prev_render_neck = None

        # build transformer architecture.
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        # set loss weight.
        self.loss_weight = np.array(loss_weight)
        assert self.loss_weight.shape[-1] == 1

        # set evaluation configurations.
        self.eval_within_grid = eval_within_grid
        self._init_layers()
        
        if self.sem_norm and not self.turn_on_flow: # occ pred_head
            sem_raymarching_branch = []
            for _ in range(self.num_pred_fcs):
                sem_raymarching_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
                sem_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                sem_raymarching_branch.append(nn.ReLU(inplace=True))
            sem_raymarching_branch.append(nn.Linear(self.embed_dims, self.num_pred_height * self.num_classes))   # C -> cls
            self.sem_raymarching_branch = nn.Sequential(*sem_raymarching_branch)

            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 32
            self.mlp_shared = nn.Sequential(
                nn.Conv3d(self.num_classes, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.reset_parameters()

        if self.obj_motion_norm and self.turn_on_flow: # flow pred_head
            flow_raymarching_branch = []
            flow_raymarching_branch.append(nn.Linear(self.embed_dims, self.num_pred_height * self.num_classes))   # C -> cls
            self.flow_raymarching_branch = nn.Sequential(*flow_raymarching_branch)

            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 32
            self.fourier_embed = Fourier_Embed_v1(nhidden)

            self.mlp_shared = nn.Sequential(
                nn.Conv3d(self.num_classes * nhidden, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.ones_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.bias)

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # BEV query for the next frame.
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # Embeds for previous frame number.
        self.prev_frame_embedding = nn.Parameter(torch.Tensor(self.memory_queue_len, self.embed_dims))
        # Embeds for CanBus information.
        # Use position & orientation information of next frame's canbus.
        if (self.use_can_bus or self.use_command or self.use_vel_steering or self.use_vel or self.use_steering or self.use_plan_traj) and self.condition_ca_add == 'add':
            can_bus_input_dim = self.action_condition_dims
        elif (self.use_can_bus or self.use_command or self.use_vel_steering or self.use_vel or self.use_steering or self.use_plan_traj) and self.condition_ca_add == 'ca':
            can_bus_input_dim = self.action_condition_dims
        if self.use_can_bus or self.use_command or self.use_vel_steering or self.use_vel or self.use_steering or self.use_plan_traj:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(can_bus_input_dim, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            if self.can_bus_norm:
                self.fusion_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.prev_frame_embedding)
            xavier_init(self.fusion_mlp, distribution='uniform', bias=0.)
        except:
            pass

    def forward_sem_norm(self, next_bev_feats):
        """
        Args:
            next_bev_feats (Tensor): inter_num, bs, hw, c
        """
        inter_num, B, HW, C = next_bev_feats.shape
        next_bev_feats = next_bev_feats.view(inter_num*B, self.bev_w, self.bev_h, C).transpose(1,2)    # inter_num*B, H,W,C

        # 1. obtain semantic prediction.
        sem_pred = self.sem_raymarching_branch(next_bev_feats)  # inter_num*B, H,W, D*cls
        sem_pred = sem_pred.view(*sem_pred.shape[:-1], self.num_pred_height, self.num_classes) # inter*B,H,W,D,cls 
        sem_label = torch.argmax(sem_pred.detach(), dim=-1)     # inter*B,H,W,D
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(sem_label[0,0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        sem_code = F.one_hot(sem_label.long(), num_classes=self.num_classes).float().permute(0,4,1,2,3).contiguous() # inter*B,cls,H,W,D

        # 2. generate parameter-free normalized activations
        next_bev_feats = self.param_free_norm(next_bev_feats)
        next_bev_feats = next_bev_feats.view(*next_bev_feats.shape[:-1], self.num_pred_height, -1)  # inter*B,H,W,D,C'
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(0)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()

        # 3. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(sem_code)    # inter*B,C',H,W,D
        gamma = self.mlp_gamma(actv).permute(0,2,3,4,1) # inter*B,H,W,D,C'
        beta = self.mlp_beta(actv).permute(0,2,3,4,1)   # inter*B,H,W,D,C'

        # apply scale and bias
        next_bev_feats = (gamma * next_bev_feats + beta).contiguous()
        next_bev_feats = next_bev_feats.view(inter_num, B, self.bev_h, self.bev_w, C).transpose(2,3).flatten(2,3)  # inter,B,HW,C

        sem_pred = sem_pred.view(inter_num, B, *sem_pred.shape[1:]).permute(0,1,5,2,3,4)    # inter,B,cls,H,W,D

        return next_bev_feats, sem_pred

    def forward_obj_motion_norm(self, next_bev_feats, occ_3D):
        """
        Args:
            next_bev_feats (Tensor): inter_num, bs, hw, c
            occ_3D: inter,bs,hw,d
        """
        # next_bev_feats
        inter_num, B, HW, C = next_bev_feats.shape
        next_bev_feats = next_bev_feats.view(inter_num*B, self.bev_w, self.bev_h, C).transpose(1,2)    # inter_num*B, H,W,C
        # occ_3D
        occ_3D = occ_3D.repeat(inter_num, 1, 1, 1).contiguous()
        occ_3D = occ_3D.view(inter_num*B, self.bev_w, self.bev_h, self.num_pred_height).transpose(1,2)  # inter_num*B, H,W,D

        # 1. obtain flow prediction.
        flow_pred = self.flow_raymarching_branch(next_bev_feats)  # inter_num*B, H,W, D*3
        flow_pred = flow_pred.view(*flow_pred.shape[:-1], self.num_pred_height, self.num_classes) # inter*B,H,W,D,3
        flow_label = flow_pred.detach() * (occ_3D.detach() > 0).float().unsqueeze(-1)    # inter*B,H,W,D,3
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(flow_label[0].mean(2)[...,1].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        flow_label = rearrange(flow_label, "b h w d c -> (b h w d c)")

        # 2. fourier embed
        flow_label = self.fourier_embed(flow_label)
        flow_label = rearrange(flow_label, "(b h w d c) c2 -> b h w d (c c2)", 
                            b=B, h=self.bev_h, w=self.bev_w, d=self.num_pred_height, c=self.num_classes, c2=self.fourier_embed.dim)
        flow_label = flow_label.permute(0,4,1,2,3)  # inter*B,C,H,W,D

        # 3. generate parameter-free normalized activations
        next_bev_feats = self.param_free_norm(next_bev_feats)
        next_bev_feats = next_bev_feats.view(*next_bev_feats.shape[:-1], self.num_pred_height, -1)  # inter*B,H,W,D,C'

        # 3. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(flow_label)    # inter*B,C',H,W,D
        gamma = self.mlp_gamma(actv).permute(0,2,3,4,1) # inter*B,H,W,D,C'
        beta = self.mlp_beta(actv).permute(0,2,3,4,1)   # inter*B,H,W,D,C'

        # apply scale and bias
        next_bev_feats = (gamma * next_bev_feats + beta).contiguous()
        next_bev_feats = next_bev_feats.view(inter_num, B, self.bev_h, self.bev_w, C).transpose(2,3).flatten(2,3)  # inter,B,HW,C

        flow_pred = flow_pred.view(inter_num, B, *flow_pred.shape[1:]).permute(0,1,5,2,3,4)    # inter,B,cls,H,W,D

        return next_bev_feats, flow_pred

    @auto_fp16(apply_to=('prev_features'))
    def _get_next_bev_features(self, prev_features, img_metas, target_frame_index, plan_traj, command, vel_steering,
                               tgt_points, ref_points, bev_h, bev_w, bev_sem_gts=None, flow_3D=None, occ_3D=None, future2history=None):
        """ Forward function for each frame.

        Args:
            prev_features (Tensor): BEV features from previous frames input, with
                shape of (bs, num_frames, bev_h * bev_w, embed_dim).
            img_metas: information of reference frame inputs.
                key "future_can_bus": can_bus information of future frames,
                    Note, 0 represents the reference frame.
            target_frame_index: next frame information.
                For indexing target can_bus information.
            tgt_points (Tensor): query point coordinates in target frame coordinates.
            ref_points (Tensor): query point coordinates in previous frame coordinates.
        """
        bs, num_frames, _, emebd_dim = prev_features.shape
        dtype = prev_features.dtype
        #  * BEV queries.
        bev_queries = self.bev_embedding.weight.to(dtype)  # bev_h * bev_w, bev_dims
        bev_queries = bev_queries.unsqueeze(0)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w

        action_condition = None
        if self.use_can_bus:
            #   * Can-bus information.
            cur_can_bus = [img_meta['future_can_bus'][target_frame_index] for img_meta in img_metas]
            cur_can_bus = np.array(cur_can_bus)[:, self.can_bus_dims]  # bs, 18
            cur_can_bus = torch.from_numpy(cur_can_bus).to(dtype).to(bev_pos.device)    # bs,4  (delta_x, delta_y, delta_z, delta_yaw)
            # fourier embed
            if self.condition_ca_add == 'ca':
                cur_can_bus = self.fourier_embed_canbus(cur_can_bus)
            action_condition = cur_can_bus

        elif self.use_plan_traj:
            #  * Plan Traj
            cur_can_bus = plan_traj[:, -1, :2].float() # bs, 2
            # fourier embed
            if self.condition_ca_add == 'ca':
                cur_can_bus = self.fourier_embed_plantraj(cur_can_bus)
            action_condition = cur_can_bus

        if self.use_command:
            command = command.unsqueeze(0)  # bs,1 (command)
            # fourier embed
            if self.condition_ca_add == 'ca':
                command = self.fourier_embed_command(command)
            if action_condition is None:
                action_condition = command
            else:
                action_condition = torch.cat([action_condition, command], dim=-1)

        if self.use_vel_steering:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            # fourier embed
            if self.condition_ca_add == 'ca':
                vel_steering = self.fourier_embed_velsteering(vel_steering)
            if action_condition is None:
                action_condition = vel_steering
            else:
                action_condition = torch.cat([action_condition, vel_steering], dim=-1)
        
        if self.use_vel:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            vel = vel_steering[:, :self.vel_dims]
            # fourier embed
            if self.condition_ca_add == 'ca':
                vel = self.fourier_embed_vel(vel)
            if action_condition is None:
                action_condition = vel
            else:
                action_condition = torch.cat([action_condition, vel], dim=-1)
        
        if self.use_steering:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            steering = vel_steering[:, -1:]
            # fourier embed
            if self.condition_ca_add == 'ca':
                steering = self.fourier_embed_steering(steering)
            if action_condition is None:
                action_condition = steering
            else:
                action_condition = torch.cat([action_condition, steering], dim=-1)
        
        if action_condition is not None:
            action_condition = self.fusion_mlp(action_condition)

        #  * sum different query embedding together.
        #    (bs, bev_h * bev_w, dims)
        if (self.use_can_bus or self.use_plan_traj) and self.condition_ca_add == 'add':
            bev_queries_input = bev_queries + action_condition.unsqueeze(1)
        else:
            bev_queries_input = bev_queries

        # 2. obtain prev embeddings (bs, num_frames, bev_h * bev_w, dims).
        if self.prev_render_neck:
            render_dict = self.prev_render_neck(prev_features.view(bs, num_frames, bev_w, bev_h, emebd_dim).transpose(2,3), 
                                                bev_sem_gts, flow_3D, occ_3D, future2history)
            prev_features = render_dict['bev_embed']
            bev_occ_pred = render_dict['bev_occ_pred']  # B,D,H,W
            san_saw_output = render_dict['san_saw_output']
            if not self.turn_on_flow:
                bev_sem_pred = render_dict['bev_sem_pred']  # B,cls,H,W

        frame_embedding = self.prev_frame_embedding
        prev_features_input = (prev_features +
                               frame_embedding[None, :, None, :])

        # 3. do transformer layers to get BEV features.
        next_bev_feat = self.transformer(
            prev_features_input,    # B, n_frame, h*w, c
            bev_queries_input,      # B, h*w, c
            tgt_points=tgt_points,  # B, h*w, 2
            ref_points=ref_points,  # B, h*w, history_num, 2
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
            action_condition=action_condition,
        )  # inter_num, bs, bev_h*bev_w, embed_dims

        # 4. sem_norm
        if self.sem_norm and not self.turn_on_flow:
            next_bev_feat, bev_sem_pred = self.forward_sem_norm(next_bev_feat)
        # 4. obj_motion_norm
        if self.obj_motion_norm and self.turn_on_flow:
            next_bev_feat_norm, bev_sem_pred = self.forward_obj_motion_norm(next_bev_feat.clone(), occ_3D)
            next_bev_feat = next_bev_feat_norm
        elif self.turn_on_flow:
            bev_sem_pred = None

        return next_bev_feat, bev_occ_pred, bev_sem_pred, san_saw_output

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self,
                prev_feats,
                img_metas,
                target_frame_index,
                plan_traj,
                command,
                vel_steering,
                tgt_points,  # tgt_points config for self-attention.
                ref_points,  # ref_points config for cross-attention.
                bev_h, bev_w,
                bev_sem_gts=None,
                flow_3D=None,
                occ_3D=None,
                future2history=None):
        f"""Forward function: a wrapper function for self._get_next_bev_features
        
        From previous multi-frame BEV features (mlvl_feats) predict 
        the next-frame of point cloud.
        Args:
            mlvl_feats (Tensor): BEV features from previous frames input, with
                shape of (bs, num_frames, bev_h * bev_w, embed_dim).
            img_metas: information of reference frame inputs.
                key "future_can_bus": can_bus information of future frames,
                    Note, 0 represents the reference frame.
            tgt_points (Tensor): query point coordinates in target frame coordinates.
            ref_points (Tensor): query point coordinates in previous frame coordinates.
        Returns:
            A dict with:
                next_bev_feature (Tensor): prediction of BEV features of next frame.
                next_bev_pred (Tensor): prediction of next BEV occupancy (Freespace).
        """
        bs, num_frames, bev_grids_num, bev_dims = prev_feats.shape
        assert bev_dims == self.embed_dims
        assert bev_h * bev_w == bev_grids_num
        assert bev_h * bev_w == tgt_points.shape[1]

        next_bev_feat, bev_occ_pred, bev_sem_pred, san_saw_output = self._get_next_bev_features(
            prev_feats, img_metas, target_frame_index, plan_traj, command, vel_steering,
            tgt_points, ref_points, bev_h, bev_w, bev_sem_gts, flow_3D, occ_3D, future2history)
        return next_bev_feat, bev_occ_pred, bev_sem_pred, san_saw_output

    def forward_head(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num, inter_num, bs, bev_h * bev_w, dims]
        """
        pass

    def _process_gt_points(self, bev_preds, gt_points, batched_origin_points,
                           valid_frames, start_idx, pred_frame_num,
                           bev_h, bev_w, pc_range):
        """Pre-process ground-truth point clouds.

        From gt_points, select those from valid_frames.
        Args:
            gt_points: A list (batch) of tensor with last dimension as [x, y, z, ..., t_index]
        Returns:
            batched_origin_points: A tensor with shape as [b, num_frame, 3]
            batched_gt_grids: A tensor with shape as [b, -1, 3]
            batched_gt_tindex: A tensor with shape as [b, -1]
        """
        valid_frame_num, inter_num, bs, token_num, num_height_pred = bev_preds.shape

        # Pre-process groundtruth.
        max_pts_num = 0
        batched_gt_points = []
        for b in range(bs):
            cur_gt_points = gt_points[b]
            valid_gt_points = []
            for i in range(start_idx, pred_frame_num):
                if i not in valid_frames:
                    # only compute loss for frames in valid_frames.
                    continue
                chosen_idx = (cur_gt_points[:, -1] == i)
                valid_gt_points.append(cur_gt_points[chosen_idx])
            valid_gt_points = torch.cat(valid_gt_points, 0)
            batched_gt_points.append(valid_gt_points)
            max_pts_num = max(max_pts_num, valid_gt_points.shape[0])

        # bs, num_frame, 3
        if batched_origin_points is None:
            batched_origin_points = torch.from_numpy(
                np.zeros((bs, len(valid_frames), 3))).to(bev_preds.dtype).to(bev_preds.device)
        batched_gt_points = torch.stack([
            F.pad(item, (0, 0, 0, max_pts_num - len(item)), mode='constant', value=float('nan'), )
            for item in batched_gt_points
        ])
        batched_gt_tindex = batched_gt_points[..., -1].contiguous()  # bs, max_pts_num
        batched_gt_tindex = batched_gt_tindex - start_idx  # tindex should start from 0.
        batched_gt_tindex[torch.isnan(batched_gt_tindex)] = -1

        batched_gt_points = batched_gt_points[..., :3].contiguous()  # bs, max_pts_num, 3

        # voxelize the origin_points and gt_points.
        batched_origin_grids = e2e_predictor_utils.coords_to_voxel_grids(
            batched_origin_points, bev_h=bev_h, bev_w=bev_w,
            pillar_num=num_height_pred, pc_range=pc_range)
        batched_gt_grids = e2e_predictor_utils.coords_to_voxel_grids(
            batched_gt_points, bev_h=bev_h, bev_w=bev_w,
            pillar_num=num_height_pred, pc_range=pc_range)

        # limit the tindex to not larger than valid_frame_num.
        batched_gt_tindex = torch.clamp(batched_gt_tindex, max=valid_frame_num - 1)
        return (batched_origin_grids, batched_origin_points,
                batched_gt_grids, batched_gt_points,
                batched_gt_tindex)

    @force_fp32(apply_to=('pred_dict'))
    def loss(self,
             pred_dict,
             gt_points,
             start_idx,
             tgt_bev_h,
             tgt_bev_w,
             tgt_pc_range,
             pred_frame_num,
             img_metas=None,
             batched_origin_points=None):
        """"Loss function.
        Args:
            pred_dict: A dictionary maintaining the point cloud prediction of different frames.
            gt_points: A list of concantenated point clouds.
                The size of list represents {bs}, the last dimension of point clouds
                indicate the timestamp of different frame.
            start_idx: the first item in pred_dict represents of which frame.
                If pred_cur_frame, start_idx should be 0;
                else start_idx should be 1;
            img_metas:
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pass

    @force_fp32(apply_to=('pred_dict'))
    def get_point_cloud_prediction(
            self,
            pred_dict,
            gt_points,
            start_idx,
            tgt_bev_h,
            tgt_bev_w,
            tgt_pc_range,
            img_metas=None,
            batched_origin_points=None):
        """"Loss function.
        Args:
            pred_dict: A dictionary maintaining the point cloud prediction of different frames.
            gt_points: A list of concantenated point clouds.
                The size of list represents {bs}, the last dimension of point clouds
                indicate the timestamp of different frame.
            start_idx: the first item in pred_dict represents of which frame.
                If pred_cur_frame, start_idx should be 0;
                else start_idx should be 1;
            img_metas:
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pass

    @force_fp32(apply_to=('pred_dict'))
    def get_point_cloud_prediction(self,
                                   pred_dict,
                                   gt_points,
                                   start_idx,
                                   bev_h,
                                   bev_w,
                                   pc_range,
                                   img_metas=None,
                                   batched_origin_points=None):
        """ Decode predictions to generate point clouds.
        """
        pass

    def get_rendered_pcds(self,
                          origin,
                          points,
                          tindex,
                          gt_dist,
                          pred_dist,
                          pc_range,):
        """Generate rendered point cloud from the predicted occupancy.

        Args:
            origin: the origin point of target point cloud. (b, num_frames, 3)
            points: ground-truth point at future frame, used for determining direction. (b, -1, 3)
            tindex: timestamps of different frames and point clouds. (b, -1)

            gt_dist: used for masking useless points. (b, -1)
            pred_dist: decoded distance from 3D occupancy prediction. (b, -1)

            pc_range: to limit point cloud prediction if eval_within_grid.
        """
        bs, num_frames, _ = origin.shape
        pcds = []
        for b in range(bs):
            cur_origin = origin[b]  # num_frame, 3
            cur_points = points[b]  # -1, 3
            cur_tindex = tindex[b]  # -1

            cur_gt_dist = gt_dist[b]  # -1
            cur_pred_dist = pred_dist[b]  # -1

            cur_batch_pcds = []
            for t in range(num_frames):
                mask = torch.logical_and(cur_tindex == t, cur_gt_dist > 0.)
                if self.eval_within_grid:
                    mask = torch.logical_and(
                        mask, e2e_predictor_utils.get_inside_mask(cur_points, pc_range))

                cur_points_t = cur_points[mask]
                r = cur_points_t - cur_origin[t].view(1, 3)
                r_norm = r / torch.sqrt((r ** 2).sum(1, keepdims=True))

                cur_pred_dist_t = cur_pred_dist[mask]
                pred_pts = cur_origin[t].view(1, 3) + r_norm * cur_pred_dist_t.view(-1, 1)

                cur_batch_pcds.append(pred_pts)
            pcds.append(cur_batch_pcds)
        return pcds


@HEADS.register_module()
class ViDARHeadBase(ViDARHeadTemplate):

    def __init__(self,
                 # point cloud rendering configurations.
                 ray_grid_num=1026,  # keep the same setting as 4d occupancy.
                 ray_grid_step=1.0,

                 # loss functions.
                 use_ce_loss=True,
                 use_dist_loss=False,

                 use_dense_loss=True,
                 dense_loss_weight=1.0,

                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.ray_grid_num = ray_grid_num
        self.ray_grid_step = ray_grid_step

        self.use_ce_loss = use_ce_loss
        self.use_dist_loss = use_dist_loss
        self.use_dense_loss = use_dense_loss
        assert (self.use_ce_loss or self.use_dist_loss or self.use_dense_loss)
        self.dense_loss_weight = dense_loss_weight

    def _get_grid_features(self,
                           batched_origin_grids,
                           batched_gt_grids,
                           batched_gt_tindex,
                           intermediate_sigma,
                           loss_weights,
                           ray_grid_step,
                           return_as_batch=False,):
        """Sample waypoints along each ray, and grid_sample their features from predicted volume.
        """
        bs, valid_frame_num, num_height_pred, tgt_bev_h, tgt_bev_w = intermediate_sigma[0].shape

        r_mask_total, r_feat_total, r_loss_weight_total, r_grid_length = [], [], [], []  # some returns.
        r_grids = torch.from_numpy(np.arange(0, self.ray_grid_num) + 0.5).to(
            batched_gt_grids.dtype).to(batched_gt_grids.device) * ray_grid_step
        for bs_idx in range(bs):
            for frame_idx in range(valid_frame_num):
                # get gt_grids of current batch and frame.
                cur_origin_grids = batched_origin_grids[bs_idx, frame_idx:frame_idx + 1]  # 1, 3
                cur_tindex = batched_gt_tindex[bs_idx]  # -1
                cur_gt_grids = batched_gt_grids[bs_idx]  # -1, 3
                cur_gt_grids = cur_gt_grids[cur_tindex == frame_idx]  # -1, 3

                # compute ray directions.
                cur_r = cur_gt_grids - cur_origin_grids  # -1, 3
                cur_r_norm = cur_r / torch.sqrt((cur_r ** 2).sum(-1, keepdims=True))  # -1, 3

                # compute waypoints according to directions.
                cur_r_grids = (cur_origin_grids.view(-1, 1, 3) +
                               cur_r_norm.view(-1, 1, 3) * r_grids.view(1, -1, 1))
                # add gt_points.
                cur_r_grids = torch.cat([cur_gt_grids.view(-1, 1, 3), cur_r_grids], 1)
                cur_r_length = torch.sqrt(((cur_r_grids - cur_origin_grids.view(-1, 1, 3)) ** 2).sum(-1))

                # norm to [-1, 1] to use F.grid_sample.
                cur_r_grids[..., 0] = cur_r_grids[..., 0] / tgt_bev_w
                cur_r_grids[..., 1] = cur_r_grids[..., 1] / tgt_bev_h
                cur_r_grids[..., 2] = cur_r_grids[..., 2] / num_height_pred
                cur_r_grids = cur_r_grids * 2 - 1  # points_num, grids_num, 3

                # inside boundary mask.
                cur_r_mask = ((cur_r_grids <= -1.) | (cur_r_grids >= 1)).any(-1)

                # remove rays whose gt-points out-of-3D-volume-range.
                cur_valid_mask = ((cur_r_grids[:, 0] > -1.) & (cur_r_grids[:, 0] < 1.)).all(-1)
                cur_r_mask = cur_r_mask[cur_valid_mask]
                cur_r_length = cur_r_length[cur_valid_mask]
                cur_r_grids = cur_r_grids[cur_valid_mask]

                # grid_sample waypoints features.
                all_lvl_r_feats = []
                all_lvl_r_loss_weights = []
                for lvl_idx, lvl_i_sigma in enumerate(intermediate_sigma):
                    lvl_i_sigma = lvl_i_sigma[bs_idx, frame_idx]  # num_height, bev_h, bev_w
                    lvl_i_r_sigma = F.grid_sample(
                        lvl_i_sigma.view(1, 1, *lvl_i_sigma.shape),
                        cur_r_grids.view(1, 1, *cur_r_grids.shape))
                    lvl_i_r_sigma = lvl_i_r_sigma.squeeze(0).squeeze(0).squeeze(0)
                    all_lvl_r_feats.append(lvl_i_r_sigma)

                    cur_loss_weight = loss_weights[frame_idx, lvl_idx]
                    lvl_i_r_loss_weight = lvl_i_r_sigma.new_ones(lvl_i_r_sigma.shape[0]) * cur_loss_weight
                    all_lvl_r_loss_weights.append(lvl_i_r_loss_weight)

                # inter_num, num_points, num_grids
                all_lvl_r_feats = torch.stack(all_lvl_r_feats, 0)
                # inter_num, num_points
                all_lvl_r_loss_weights = torch.stack(all_lvl_r_loss_weights, 0)

                r_mask_total.append(cur_r_mask)  # points_num, grid_num.
                r_feat_total.append(all_lvl_r_feats)  # lvl_num, points_num, grid_num.
                r_loss_weight_total.append(all_lvl_r_loss_weights)  # lvl_num, points_num
                r_grid_length.append(cur_r_length)

        r_mask_total = torch.cat(r_mask_total, 0)  # total_points_num, grid_num
        r_feat_total = torch.cat(r_feat_total, 1)  # lvl_num, total_points_num, grid_num.
        r_loss_weight_total = torch.cat(r_loss_weight_total, 1)  # lvl_num, total_points_num.
        r_grid_length = torch.cat(r_grid_length, 0)  # total_points_num, grid_num

        r_mask_total = r_mask_total.float().masked_fill(r_mask_total.bool(), float('-inf'))
        r_feat_total = r_feat_total + r_mask_total.unsqueeze(0)

        if return_as_batch:
            r_mask_total = r_mask_total.view(bs, -1, *r_mask_total.shape[1:])
            r_feat_total = r_feat_total.view(len(intermediate_sigma), bs, -1, *r_feat_total.shape[2:])
            r_loss_weight_total = r_loss_weight_total.view(
                len(intermediate_sigma), bs, -1, *r_loss_weight_total.shape[2:])
            r_grid_length = r_grid_length.view(bs, -1, *r_grid_length.shape[1:])

        return r_mask_total, r_feat_total, r_loss_weight_total, r_grid_length

    @force_fp32(apply_to=('pred_dict'))
    def loss(self,
             pred_dict,
             gt_points,
             start_idx,
             tgt_bev_h,
             tgt_bev_w,
             tgt_pc_range,
             pred_frame_num,
             img_metas=None,
             batched_origin_points=None,
             loss_weight=None):
        """"Loss function.
        Args:
            pred_dict: A dictionary maintaining the point cloud prediction of different frames.
            gt_points: A list of concantenated point clouds.
                The size of list represents {bs}, the last dimension of point clouds
                indicate the timestamp of different frame.  # bs*[N,4  xyzt] 
            start_idx: the first item in pred_dict represents of which frame.
                If pred_cur_frame, start_idx should be 0;
                else start_idx should be 1;
            img_metas:
            batched_origin_points: bs,Lout,3
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Pre-process predicted results.
        valid_frames = pred_dict['valid_frames']
        bev_preds = pred_dict['next_bev_preds'] # Lout, inter_num, bs, h*w, d  每一帧的前后i帧

        bev_preds = bev_preds[:, -1:]  # only select the last feature as outputs.
        valid_frame_num, inter_num, bs, token_num, num_height_pred = bev_preds.shape
        sigma = bev_preds

        # Pre-process groundtruth.
        (batched_origin_grids, batched_origin_points,
         batched_gt_grids, batched_gt_points,
         batched_gt_tindex) = self._process_gt_points(
            sigma, gt_points, batched_origin_points,
            valid_frames, start_idx, pred_frame_num,
            tgt_bev_h, tgt_bev_w, tgt_pc_range
        )
        # batched_origin_grids, batched_origin_points: B,Lout,3      每帧原点的体素坐标、原坐标
        # batched_gt_grids, batched_gt_points:         B,Lout,N,3    每帧点云的体素坐标、原坐标
        # batched_gt_tindex:                           B,Lout        每帧点对应的frame_t

        # compute directional orientation vector to each point.
        intermediate_sigma = [] # inter_num,[bs, Lout, d,h,w]
        for i in range(inter_num):
            # valid_frame_num, bs, bev_h * bev_w, num_height_pred
            # --> bs, valid_frame_num, num_height, bev_h * bev_w
            # --> bs, valid_frame_num, num_height, bev_h, bev_w
            lvl_i_sigma = sigma[:, i]
            lvl_i_sigma = lvl_i_sigma.permute(1, 0, 3, 2).contiguous()
            lvl_i_sigma = lvl_i_sigma.view(
                bs, valid_frame_num, num_height_pred, tgt_bev_h, tgt_bev_w)
            intermediate_sigma.append(lvl_i_sigma)

        ray_grid_step = self.ray_grid_step
        loss_weight = self.loss_weight if loss_weight is None else loss_weight
        r_mask_total, r_feat_total, r_loss_weight_total, r_grid_length = self._get_grid_features(
            batched_origin_grids, batched_gt_grids, batched_gt_tindex,
            intermediate_sigma, loss_weight,
            ray_grid_step=ray_grid_step)

        loss_dict = dict()
        # compute loss
        r_label = r_mask_total.new_zeros(*r_feat_total.shape[:-1]).long()  # lvl_num, total_points_num
        if self.use_dist_loss:
            r_mask_total = r_mask_total.float().masked_fill(r_mask_total.bool(), float('-inf'))
            r_dist_feat = r_feat_total + r_mask_total
            r_pred_dist = self._custom_gumbel_softmax_distance(r_dist_feat, r_grid_length[None])

            scale_factor = (tgt_pc_range[3] - tgt_pc_range[0]) / tgt_bev_w
            dist_loss = (torch.abs(r_pred_dist - r_grid_length[None, :, 0]) * scale_factor)  # lvl_num, total_points_num

            dist_loss = ((dist_loss * r_loss_weight_total).sum() /
                         torch.clamp(r_loss_weight_total.sum(), min=1))
            loss_dict.update({f'dist.loss': dist_loss})
        if self.use_ce_loss:
            r_feat_total = r_feat_total.transpose(1, 2).contiguous()  # lvl_num, grid_num, total_points_num
            r_loss = F.cross_entropy(r_feat_total, r_label, reduction='none')

            r_loss = ((r_loss * r_loss_weight_total).sum() /
                      torch.clamp(r_loss_weight_total.sum(), min=1))
            loss_dict.update({f'regularization.loss': r_loss})

        if self.use_dense_loss:
            # currently, only supervise the last intermediate layer
            #  for saving computational cost.
            lvl_tgt_sigma = sigma[:, -1]
            # valid_frame_num, bs, bev_h * bev_w, num_height_pred
            # --> bs, valid_frame_num, num_height, bev_h * bev_w
            # --> bs, valid_frame_num, num_height, bev_h, bev_w
            lvl_tgt_sigma = lvl_tgt_sigma.permute(1, 0, 3, 2)
            lvl_tgt_sigma = lvl_tgt_sigma.reshape(
                bs, valid_frame_num, num_height_pred, tgt_bev_h, tgt_bev_w).contiguous()

            # get voxel gt grids.
            dense_sample_interval = 4
            voxel_grids = e2e_predictor_utils.get_bev_grids_3d(
                tgt_bev_h // dense_sample_interval, tgt_bev_w // dense_sample_interval,
                num_height_pred // dense_sample_interval, bs=bs)
            voxel_grids[..., 0] = voxel_grids[..., 0] * tgt_bev_w
            voxel_grids[..., 1] = voxel_grids[..., 1] * tgt_bev_h
            voxel_grids[..., 2] = voxel_grids[..., 2] * num_height_pred
            voxel_grids = voxel_grids.view(bs, -1, 3)
            voxel_tindex = voxel_grids.new_ones(*voxel_grids.shape[:2]).to(batched_gt_tindex.dtype)

            voxel_grids = torch.cat([voxel_grids for i in range(valid_frame_num)], 1)
            voxel_tindex = torch.cat([voxel_tindex * i for i in range(valid_frame_num)], 1)

            dense_mask_total, dense_feat_total, dense_weight_total, dense_grid_length = self._get_grid_features(
                batched_origin_grids, voxel_grids, voxel_tindex, [lvl_tgt_sigma],
                loss_weights=np.array([[1]] * valid_frame_num), return_as_batch=True,
                ray_grid_step=ray_grid_step)
            dense_feat_total = dense_feat_total[0][..., 1:].contiguous()  # bs, -1, grid_num
            dense_grid_length = dense_grid_length[..., 1:].contiguous()  # bs, -1, grid_num

            dense_dist = self._custom_gumbel_softmax_distance(
                dense_feat_total, dense_grid_length)
            voxel_pcd = self.get_rendered_pcds(
                batched_origin_grids, voxel_grids, voxel_tindex,
                dense_dist, dense_dist, tgt_pc_range)

            dense_voxel_loss = 0
            for b_id in range(bs):
                for f_id in range(valid_frame_num):
                    cur_pred_pcd = voxel_pcd[b_id][f_id].view(1, -1, 3)
                    cur_gt_pcd = batched_gt_grids[b_id][batched_gt_tindex[b_id] == f_id].view(1, -1, 3)

                    cur_gt_pcd_mask = ((cur_gt_pcd[..., 0] < tgt_bev_w - 1) &
                                       (cur_gt_pcd[..., 0] > 0) &
                                       (cur_gt_pcd[..., 1] < tgt_bev_h - 1) &
                                       (cur_gt_pcd[..., 1] > 0) &
                                       (cur_gt_pcd[..., 2] < num_height_pred - 1) &
                                       (cur_gt_pcd[..., 2] > 0))
                    cur_gt_pcd = cur_gt_pcd.squeeze(0)[cur_gt_pcd_mask.squeeze(0)][None]

                    cur_pred_pcd = cur_pred_pcd - batched_origin_grids[b_id, f_id:f_id + 1]
                    cur_gt_pcd = cur_gt_pcd - batched_origin_grids[b_id, f_id:f_id + 1]

                    # scale the loss function
                    cur_pred_pcd = cur_pred_pcd * 0.1
                    cur_gt_pcd = cur_gt_pcd * 0.1
                    if cur_gt_pcd.shape[1] == 0:
                        continue
                    loss_src, loss_tgt, _, _ = chamfer_distance(cur_pred_pcd, cur_gt_pcd)
                    dense_voxel_loss = dense_voxel_loss + (
                            (loss_src + loss_tgt) / 2.) * loss_weight[f_id, 0]

            dense_voxel_loss = dense_voxel_loss / (loss_weight.sum() * bs)
            loss_dict.update({'loss.dense_voxel': dense_voxel_loss * self.dense_loss_weight})
        return loss_dict

    @force_fp32(apply_to=('pred_dict'))
    def get_point_cloud_prediction(self,
                                   pred_dict,
                                   gt_points,
                                   start_idx,
                                   tgt_bev_h,
                                   tgt_bev_w,
                                   tgt_pc_range,
                                   img_metas=None,
                                   batched_origin_points=None):
        """ Rewrite the function of get_point_cloud_prediction.

        After finding the path, we use softmax + max_index to find the index with largest
            probability. And use its distance as the final results.
        """
        # Pre-process predicted results.
        bev_preds = pred_dict['next_bev_preds']
        valid_frames = pred_dict['valid_frames']
        pred_frame_num, inter_num, bs, token_num, num_height_pred = bev_preds.shape
        sigma = bev_preds[:, -1]  # get the last prediction after transformer layers.

        # Pre-process groundtruth.
        (batched_origin_grids, batched_origin_points,
         batched_gt_grids, batched_gt_points,
         batched_gt_tindex) = self._process_gt_points(
            bev_preds, gt_points, batched_origin_points,
            valid_frames, start_idx, pred_frame_num,
            tgt_bev_h, tgt_bev_w, tgt_pc_range
        )

        # frame_num, bs, bev_h * bev_w, num_height_pred
        # ---> bs, frame_num, num_height, bev_h * bev_w
        sigma = sigma.permute(1, 0, 3, 2).contiguous()
        # ---> bs, frame_num, num_height, bev_h, bev_w
        sigma = sigma.view(bs, pred_frame_num, num_height_pred, tgt_bev_h, tgt_bev_w)

        gt_dist = batched_gt_grids.new_zeros(*batched_gt_grids.shape[:2])
        pred_dist = torch.zeros_like(gt_dist)
        r_grids = torch.from_numpy(np.arange(0, self.ray_grid_num) + 0.5).to(
            batched_gt_grids.dtype).to(batched_gt_grids.device) * self.ray_grid_step
        for bs_idx in range(bs):
            for frame_idx in range(pred_frame_num):
                cur_origin_grids = batched_origin_grids[bs_idx, frame_idx:frame_idx + 1]  # 1, 3
                cur_tindex = batched_gt_tindex[bs_idx]  # -1

                # Parse ground-truth.
                cur_gt_grids = batched_gt_grids[bs_idx][cur_tindex == frame_idx]  # -1, 3
                if len(cur_gt_grids) == 0: continue
                gt_dist[bs_idx, cur_tindex == frame_idx] = torch.sqrt(((cur_gt_grids - cur_origin_grids) ** 2).sum(-1))

                # Parse predictions.
                cur_r = cur_gt_grids - cur_origin_grids  # -1, 3
                cur_r_norm = cur_r / torch.sqrt((cur_r ** 2).sum(-1, keepdims=True))  # -1, 3

                cur_r_grids = (cur_origin_grids.view(-1, 1, 3) +
                               cur_r_norm.view(-1, 1, 3) * r_grids.view(1, -1, 1))
                cur_r_length = torch.sqrt(((cur_r_grids - cur_origin_grids.view(-1, 1, 3)) ** 2).sum(-1))

                cur_r_grids[..., 0] = cur_r_grids[..., 0] / tgt_bev_w
                cur_r_grids[..., 1] = cur_r_grids[..., 1] / tgt_bev_h
                cur_r_grids[..., 2] = cur_r_grids[..., 2] / num_height_pred
                cur_r_grids = cur_r_grids * 2 - 1  # points_num, grids_num, 3

                cur_sigma = sigma[bs_idx, frame_idx]  # num_height, bev_h, bev_w
                cur_sigma = F.grid_sample(cur_sigma.view(1, 1, *cur_sigma.shape),
                                          cur_r_grids.view(1, 1, *cur_r_grids.shape))
                cur_sigma = cur_sigma.float().masked_fill((cur_sigma == 0), float('-inf'))

                cur_sigma = cur_sigma.squeeze(0).squeeze(0).squeeze(0)  # points_num, grids_num
                _, max_sigma_idx = cur_sigma.max(1)  # points_num
                cur_dist = torch.gather(cur_r_length, dim=1, index=max_sigma_idx.view(-1, 1)).squeeze(-1)

                pred_dist[bs_idx, cur_tindex == frame_idx] = cur_dist

        scale_factor = (tgt_pc_range[3] - tgt_pc_range[0]) / tgt_bev_w
        pred_dist = pred_dist * scale_factor
        gt_dist = gt_dist * scale_factor
        # 2. render pred_point_cloud and gt_point_cloud.
        pred_pcds = self.get_rendered_pcds(
            batched_origin_points, batched_gt_points, batched_gt_tindex,
            gt_dist, pred_dist, tgt_pc_range)
        gt_pcds = self.get_rendered_pcds(
            batched_origin_points, batched_gt_points, batched_gt_tindex,
            gt_dist, gt_dist, tgt_pc_range)

        decode_dict = dict(
            pred_pcds=pred_pcds,
            gt_pcds=gt_pcds,
            origin=batched_origin_points,
        )
        return decode_dict

    def _custom_gumbel_softmax_distance(self, grid_embed, grid_length):
        """ Decode differentiable distance from grid_embed and grid_length. """
        # 1. randomly sample the current ground-truth according to gumbel_sample.
        pred_dist = F.gumbel_softmax(grid_embed, hard=True)
        pred_dist = (pred_dist * grid_length).sum(-1).detach()  # bs, -1

        # 2. compute the next softmax for optimization.
        # ---- whole exponential.
        grid_embed = grid_embed - grid_embed.max(-1, keepdims=True)[0]
        exp_embed = torch.exp(grid_embed)
        exp_whole = exp_embed.sum(-1)
        # ---- next exponential.
        next_ind = (grid_length > pred_dist.unsqueeze(-1)).float()
        exp_next = (exp_embed * next_ind).sum(-1)
        prob_next = exp_next / exp_whole

        # 3. obtain the differentiable distance.
        prob_next = 1 - prob_next.detach() + prob_next
        pred_dist = prob_next * pred_dist
        return pred_dist
