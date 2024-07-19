#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
<V1.multiframe> of ViDAR future prediction head:
    * Predict future & history frames simultaneously.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mmdet.models import HEADS, build_loss

from mmcv.runner import force_fp32, auto_fp16
from .vidar_head_base import ViDARHeadBase
from projects.mmdet3d_plugin.bevformer.losses.semkitti_loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss, Smooth_L1_loss, BCE_loss
from projects.mmdet3d_plugin.bevformer.losses.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.bevformer.modules.SAN_SAW import SAN, SAW


@HEADS.register_module()
class ViDARHeadV1(ViDARHeadBase):
    def __init__(self,
                 history_queue_length,
                 soft_weight,
                 san_saw=False,
                 pred_history_frame_num=0,
                 pred_future_frame_num=0,
                 per_frame_loss_weight=(1.0,),
                 loss_weight_cfg=None,

                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.san_saw = san_saw
        self.history_queue_length = history_queue_length    # 4

        self.pred_history_frame_num = pred_history_frame_num    # 0(mem-e)
        self.pred_future_frame_num = pred_future_frame_num      # 0(mem-e)

        self.pred_frame_num = 1 + self.pred_history_frame_num + self.pred_future_frame_num
        self.per_frame_loss_weight = per_frame_loss_weight
        assert len(self.per_frame_loss_weight) == self.pred_frame_num

        self.class_weights = np.ones((self.num_classes,))
        self.class_weights[1:] = 5
        self.class_weights = torch.from_numpy(self.class_weights)

        # voxel sem losses
        if loss_weight_cfg is None:
            self.multi_loss = False
            self.loss_voxel_ce_weight = 1.0
        else:
            self.multi_loss = True
            self.loss_voxel_ce_weight = loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
            self.loss_voxel_sem_scal_weight = loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
            self.loss_voxel_geo_scal_weight = loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
            self.loss_voxel_lovasz_weight = loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)

        self.soft_weight = soft_weight
        self._init_bev_pred_layers()

        self.num_points_sampling_feat = self.transformer.decoder.num_layers
        if self.soft_weight:
            self.bev_soft_weights = nn.Sequential(
                nn.Linear(self.embed_dims//2, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims//2, self.num_points_sampling_feat),
            )

            self.occ_pred_conv = nn.Sequential(
                nn.Linear(self.embed_dims//2, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims//2, self.num_pred_height * self.num_classes * self.pred_frame_num)
            )
            if self.turn_on_flow:
                self.occ_pred_conv[3].bias = nn.parameter.Parameter(torch.zeros(self.num_pred_height * self.num_classes * self.pred_frame_num).float(), requires_grad=True)

        if self.san_saw and not self.turn_on_flow:  # occ pred_head
            self.classifier = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1, stride=1, bias=True)

            self.selecte_classes = [1,2,3,4,5,6,7,8]
            self.san_stage = SAN(inplanes=self.embed_dims, selected_classes=self.selecte_classes)
            # self.saw_stage = SAW(dim=self.embed_dims, selected_classes=self.selecte_classes, relax_denom=2.0, classifier=self.classifier)

    def forward_san_saw(self, next_bev_feats):
        """
        Args:
            next_bev_feats (Tensor): Lout, inter_num, bs, hw, c
        """
        next_bev_feats = next_bev_feats.permute(1,2,0,4,3)  # inter_num,bs,Lout,c,hw
        inter_num, B, L, C, _ = next_bev_feats.shape
        next_bev_feats = next_bev_feats.view(inter_num, B*L, C, self.bev_w, self.bev_h).transpose(3,4)    # inter_num, B*L, C, h,w

        embed_ori = next_bev_feats  # inter_num,B*L,C,h,w

        # 1. classfier
        embed = next_bev_feats.flatten(0,1) # inter_num*B*L, C,H,W
        sem_pred = self.classifier(embed.detach())  # inter_num*B*L,cls,H,W
        
        # 2. SAN
        embed = self.san_stage(embed, sem_pred) # inter_num*B*L, C,H,W
        embed_san = embed.view(inter_num, B*L, C, self.bev_h, self.bev_w)   # inter_num, B*L, C,H,W

        # 3. SAW
        # saw_loss = self.saw_stage(embed)

        embed = embed.transpose(2,3)    # inter_num*B*L,C,W,H
        embed = embed.reshape(inter_num, B, L, C, -1)
        embed = embed.permute(2, 0, 1, 4, 3).contiguous()  # L, inter_num, B, HW, C
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(embed[0].max(-1)[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        if self.training:
            out_dict = {
                'sem_pred': sem_pred.view(inter_num, B*L, -1, self.bev_h, self.bev_w),   # inter_num,B*L,cls,H,W
                'embed_ori': embed_ori, # inter_num,B*L,C,H,W
                'embed_san': embed_san, # inter_num,B*L,C,H,W
                # 'saw_loss': torch.tensor([0]).to(embed),   # 1
            }
        else:
            out_dict = None
        return embed, out_dict

    def _init_bev_pred_layers(self):
        """Overwrite the {self.bev_pred_head} of super()._init_layers()
        """
        bev_pred_branch = []
        mid_dims = self.embed_dims//2 if self.soft_weight else self.embed_dims
        for _ in range(self.num_pred_fcs):
            bev_pred_branch.append(nn.Linear(self.embed_dims, mid_dims))
            bev_pred_branch.append(nn.LayerNorm(mid_dims))
            bev_pred_branch.append(nn.ReLU(inplace=True))

        # not_soft_weight: direct output
        if not self.soft_weight:
            bev_pred_branch.append(nn.Linear(
                mid_dims, self.pred_frame_num * self.num_pred_height * self.num_classes))
            if self.turn_on_flow:
                bev_pred_branch[-1].bias = nn.parameter.Parameter(torch.zeros(self.num_pred_height * self.num_classes * self.pred_frame_num).float(), requires_grad=True)

        bev_pred_head = nn.Sequential(*bev_pred_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # Auxiliary supervision for all intermediate results.
        num_pred = self.transformer.decoder.num_layers if self.transformer.decoder.return_intermediate else 1
        self.bev_pred_head = _get_clones(bev_pred_head, num_pred)

    def forward_head_soft(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num_set, inter_num, bs, bev_h * bev_w, dims]    pred_frame_num_set = cur + future_select
                self.pred_frame_num: history frames + current frame + future frames.
        """
        next_bev_preds = []
        for lvl in range(next_bev_feats.shape[1]):
            next_bev_preds.append(self.bev_pred_head[lvl](next_bev_feats[:, lvl])) # Lout,B,hw,C
        
        if self.soft_weight:
            bev_soft_weights = self.bev_soft_weights(next_bev_preds[-1])    # Lout,B,hw,3
            bev_soft_weights = torch.softmax(bev_soft_weights, dim=1)       # Lout,B,hw,3
        else:
            bev_soft_weights = torch.ones([next_bev_preds[-1].shape[0], next_bev_preds[-1].shape[1], 1, self.num_points_sampling_feat], ).to(next_bev_preds[0].device) / self.num_points_sampling_feat
        
        # soft_weight
        out_bev_feats = 0
        for feat, weights in zip(next_bev_preds, torch.unbind(bev_soft_weights, dim=-1)):
            out_bev_feats += feat * weights.unsqueeze(-1)
        
        # out pred
        out_occ = self.occ_pred_conv(out_bev_feats) # Lout,B,hw,c -> Lout,B,hw,d*cls*pred_frame

        # base + pred
        out_occ = out_occ.view(*out_occ.shape[:-1], self.num_pred_height, self.num_classes, self.pred_frame_num)    # Lout,B,hw,d,cls,pred_frame
        base_occ = out_occ[..., self.pred_history_frame_num][..., None]  # Lout,B,hw,d,cls,1 当前帧
        out_occ = torch.cat([
            out_occ[..., :self.pred_history_frame_num] + base_occ,   # history都+当前帧
            base_occ,
            out_occ[..., self.pred_history_frame_num + 1:] + base_occ# future都+当前帧
        ], -1)                                                       # Lout,B,hw,d,cls,pred_frame

        out_occ = out_occ.permute(0, 5, 1, 2, 3, 4).contiguous().unsqueeze(1)     # Lout, inter_num=1, history+1+future,B,hw,d,cls

        return out_occ  
    
    def forward_head_layers(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [Lout, inter_num, bs, bev_h * bev_w, dims]    Lout = cur + future_select
                self.pred_frame_num: history frames + current frame + future frames.
        """
        next_bev_preds = []
        for lvl in range(next_bev_feats.shape[1]):
            #  ===> Lout, bs, h*w, d, num_frame
            next_bev_pred = self.bev_pred_head[lvl](next_bev_feats[:, lvl]) # C -> d * num_cls * num_frame 
            next_bev_pred = next_bev_pred.view(
                *next_bev_pred.shape[:-1], self.num_pred_height, self.num_classes, self.pred_frame_num)

            base_bev_pred = next_bev_pred[..., self.pred_history_frame_num][..., None]  # Lout, bs, h*w, d, num_cls, 1 当前帧
            next_bev_pred = torch.cat([
                next_bev_pred[..., :self.pred_history_frame_num] + base_bev_pred,   # history都+当前帧
                base_bev_pred,
                next_bev_pred[..., self.pred_history_frame_num + 1:] + base_bev_pred# future都+当前帧
            ], -1)  # Lout, bs, h*w, d, num_cls, history+1+future

            next_bev_pred = next_bev_pred.permute(0, 5, 1, 2, 3, 4).contiguous()   # Lout, history+1+future, bs, h*w, d, num_cls
            next_bev_preds.append(next_bev_pred)
        # Lout, inter_num, history+1+future, bs, h*w, d, num_cls
        next_bev_preds = torch.stack(next_bev_preds, 1)
        return next_bev_preds
    
    def forward_head(self, next_bev_feats):
        if self.soft_weight:
            return self.forward_head_soft(next_bev_feats)   # multi-decoder_layers soft_weight_sum
        else:
            return self.forward_head_layers(next_bev_feats) # multi-decoder_layers

    def _get_reference_gt_points(self,
                                 gt_points,
                                 src_frame_idx_list,
                                 tgt_frame_idx_list,
                                 img_metas):
        """Transform gt_points at src_frame_idx in {src_frame_idx_list} to the coordinate space
        of each tgt_frame_idx in {tgt_frame_idx_list}.
        """
        bs = len(gt_points)
        aligned_gt_points = []
        batched_origin_points = []
        for frame_idx, src_frame_idx, tgt_frame_idx in zip(
                range(len(src_frame_idx_list)), src_frame_idx_list, tgt_frame_idx_list):
            # 1. get gt_points belongs to src_frame_idx.
            src_frame_gt_points = [p[p[:, -1] == src_frame_idx] for p in gt_points]

            # 2. get transformation matrix..
            src_to_ref = [img_meta['total_cur2ref_lidar_transform'][src_frame_idx] for img_meta in img_metas]
            src_to_ref = gt_points[0].new_tensor(np.array(src_to_ref))  # bs, 4, 4
            ref_to_tgt = [img_meta['total_ref2cur_lidar_transform'][tgt_frame_idx] for img_meta in img_metas]
            ref_to_tgt = gt_points[0].new_tensor(np.array(ref_to_tgt))  # bs, 4, 4
            src_to_tgt = torch.matmul(src_to_ref, ref_to_tgt)

            # 3. transfer src_frame_gt_points to src_to_tgt.
            aligned_gt_points_per_frame = []
            for batch_idx, points in enumerate(src_frame_gt_points):
                new_points = points.clone()  # -1, 4
                new_points = torch.cat([
                    new_points[:, :3], new_points.new_ones(new_points.shape[0], 1)
                ], 1)
                new_points = torch.matmul(new_points, src_to_tgt[batch_idx])
                new_points[..., -1] = frame_idx
                aligned_gt_points_per_frame.append(new_points)
            aligned_gt_points.append(aligned_gt_points_per_frame)

            # 4. obtain the aligned origin points.
            aligned_origin_points = torch.from_numpy(
                np.zeros((bs, 1, 3))).to(src_to_tgt.dtype).to(src_to_tgt.device)
            aligned_origin_points = torch.cat([
                aligned_origin_points[..., :3], torch.ones_like(aligned_origin_points)[..., 0:1]
            ], -1)
            aligned_origin_points = torch.matmul(aligned_origin_points, src_to_tgt)
            batched_origin_points.append(aligned_origin_points[..., :3].contiguous())

        # stack points from different timestamps, and transfer to occupancy representation.
        batched_gt_points = []
        for b in range(bs):
            cur_gt_points = [
                aligned_gt_points[frame_idx][b]
                for frame_idx in range(len(src_frame_idx_list))]
            cur_gt_points = torch.cat(cur_gt_points, 0)
            batched_gt_points.append(cur_gt_points)

        batched_origin_points = torch.cat(batched_origin_points, 1)
        return batched_gt_points, batched_origin_points

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
        """"Compute loss for all history according to gt_points.

        gt_points: ground-truth point cloud in each frame.
            list of tensor with shape [-1, 5], indicating ground-truth point cloud in
            each frame.
        """
        bev_preds = pred_dict['next_bev_preds'] # Lout, intermediate_num, history+1+future, bs, h*w, d
        valid_frames = np.array(pred_dict['valid_frames'])  # [0 + future_select] 
        start_frames = (valid_frames + self.history_queue_length - self.pred_history_frame_num) # [0,1,2] + history_in - history_pred 
        tgt_frames = valid_frames + self.history_queue_length                                   # [0,1,2] + history_in

        full_prev_bev_exists = pred_dict.get('full_prev_bev_exists', True)
        if not full_prev_bev_exists:
            frame_idx_for_loss = [self.pred_history_frame_num] * self.pred_frame_num
        else:
            frame_idx_for_loss = np.arange(0, self.pred_frame_num)  # [0, 1, 2]

        loss_dict = dict()
        for idx, i in enumerate(frame_idx_for_loss):
            # 1. get the predicted occupancy of frame-i.
            cur_bev_preds = bev_preds[:, :, i, ...].contiguous()    # Lout, inter_num, bs, h*w, d  每一帧的前/后 pred_i帧

            # 2. get the frame index of current frame.
            src_frames = start_frames + i

            # 3. get gt_points belonging to cur_valid_frames.
            cur_gt_points, cur_origin_points = self._get_reference_gt_points(   # bs*[N,4  xyzt]    bs,Lout,3
                gt_points,
                src_frame_idx_list=src_frames,  # [0,1,2]
                tgt_frame_idx_list=tgt_frames,  # [1,2,3]
                img_metas=img_metas)

            # 4. compute loss.
            if i != self.pred_history_frame_num:
                # For aux history-future supervision:
                #  only compute loss for cur_frame prediction.
                loss_weight = np.array([[1]] + [[0]] * (len(self.loss_weight) - 1))
            else:
                loss_weight = self.loss_weight

            cur_loss_dict = super().loss(
                dict(next_bev_preds=cur_bev_preds,
                     valid_frames=np.arange(0, len(src_frames))),
                cur_gt_points,
                start_idx=start_idx,
                tgt_bev_h=tgt_bev_h,
                tgt_bev_w=tgt_bev_w,
                tgt_pc_range=tgt_pc_range,
                pred_frame_num=len(self.loss_weight)-1,
                img_metas=img_metas,
                batched_origin_points=cur_origin_points,
                loss_weight=loss_weight)

            # 5. merge dict.
            cur_frame_loss_weight = self.per_frame_loss_weight[i]
            cur_frame_loss_weight = cur_frame_loss_weight * (idx == i)
            for k, v in cur_loss_dict.items():
                loss_dict.update({f'frame.{idx}.{k}.loss': v * cur_frame_loss_weight})
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
        """"Generate point cloud prediction.
        """
        # pred_frame_num, inter_num, num_frame, bs, bev_h * bev_w, num_height_pred
        pred_dict['next_bev_preds'] = pred_dict['next_bev_preds'][:, :, self.pred_history_frame_num, ...].contiguous()

        valid_frames = np.array(pred_dict['valid_frames'])
        valid_gt_points, cur_origin_points = self._get_reference_gt_points(
            gt_points,
            src_frame_idx_list=valid_frames + self.history_queue_length,
            tgt_frame_idx_list=valid_frames + self.history_queue_length,
            img_metas=img_metas)
        return super().get_point_cloud_prediction(
            pred_dict=pred_dict,
            gt_points=valid_gt_points,
            start_idx=start_idx,
            tgt_bev_h=tgt_bev_h,
            tgt_bev_w=tgt_bev_w,
            tgt_pc_range=tgt_pc_range,
            img_metas=img_metas,
            batched_origin_points=cur_origin_points)

    def loss_voxel(self, output_voxels, target_voxels, tag):
        B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels, size=(H, W, D), mode='trilinear', align_corners=False)
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx    # B,H,W,D
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]                  # select_vox_num, ratio**3
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space                  # B,H,W,D,ratio**3
            target_voxels = torch.mode(target_voxels, dim=-1)[0]    # B,H,W,D 取众数  即原来一个大voxel被划分成了ratio**3个voxel，取它们的众数
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        if self.multi_loss:
            loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=empty_idx)
            loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        return loss_dict
    
    def loss_occ(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='inter_{}'.format(index)))
            
        return loss_dict
    
    def loss_sem_norm(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        inter, B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels.flatten(0,1), size=(H, W, D), mode='trilinear', align_corners=False)
            output_voxels = output_voxels.view(inter, B,C,H,W,D)
        
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx    # B,H,W,D
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]                  # select_vox_num, ratio**3
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space                  # B,H,W,D,ratio**3
            target_voxels = torch.mode(target_voxels, dim=-1)[0]    # B,H,W,D 取众数  即原来一个大voxel被划分成了ratio**3个voxel，取它们的众数
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            inter_loss = CE_ssc_loss(output_voxel, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict['loss_sem_norm_{}'.format(index)] = inter_loss

        return loss_dict

    def loss_obj_motion_norm(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        inter, B, C, H, W, D = output_voxels.shape
        tB, tC, tH, tW, tD = target_voxels.shape
        
        assert torch.isnan(output_voxels).sum().item() == 0

        # output_voxel align to target_voxel
        if H != tH:
            output_voxels = F.interpolate(output_voxels.flatten(0,1), size=(tH, tW, tD), mode='trilinear', align_corners=False)
            output_voxels = output_voxels.view(inter, B,C,tH,tW,tD)

        output_voxels = output_voxels.permute(0,1,3,4,5,2)    # inter,B*L,H,W,D,3
        target_voxels = target_voxels.permute(0,2,3,4,1)      #       B*L,H,W,D,3

        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            inter_loss = (0.5) * Smooth_L1_loss(output_voxel, target_voxels, ignore_index=255)
            loss_dict['loss_obj_motion_norm_{}'.format(index)] = inter_loss

        return loss_dict
    
    def loss_voxel_flow(self, output_voxels, target_voxels, tag):                    
        B, C, H, W, D = output_voxels.shape
        tB, tC, tH, tW, tD = target_voxels.shape
        
        assert torch.isnan(output_voxels).sum().item() == 0

        # output_voxel align to target_voxel
        if H != tH:
            output_voxels = F.interpolate(output_voxels, size=(tH, tW, tD), mode='trilinear', align_corners=False)

        output_voxels = output_voxels.permute(0,2,3,4,1)    # B,tH,tW,tD,3
        target_voxels = target_voxels.permute(0,2,3,4,1)

        loss_dict = {}

        loss_dict['loss_flow_l1_{}'.format(tag)] = (0.5) * Smooth_L1_loss(output_voxels, target_voxels, ignore_index=255)

        return loss_dict
    
    def loss_flow(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, 3, h,w,d
            target_voxels =            select_frame*bs, 3, H,W,D
        """
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel_flow(output_voxel, target_voxels,  tag='inter_{}'.format(index)))
            
        return loss_dict

    def loss_bev_occ(self, output_voxels, target_voxels):
        B, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape
        
        output_voxels = output_voxels.unsqueeze(1)
        target_voxels = target_voxels.unsqueeze(1)

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels, size=(H, W, D), mode='trilinear', align_corners=False).squeeze(1)
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx    # B,H,W,D
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]                  # select_vox_num, ratio**3
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space                  # B,H,W,D,ratio**3
            target_voxels = torch.mode(target_voxels, dim=-1)[0]    # B,H,W,D 取众数  即原来一个大voxel被划分成了ratio**3个voxel，取它们的众数
            target_voxels[target_voxels<0] = 0
            target_voxels = target_voxels.long()
        
        # breakpoint()
        # import matplotlib.pyplot as plt
        # target_voxels_vis, _ = torch.max(target_voxels, dim=-1)
        # plt.imshow(target_voxels_vis[0].detach().cpu().numpy())
        # plt.show()
        
        loss_bev_occ = BCE_loss(output_voxels, target_voxels)
        return loss_bev_occ
    
    def loss_bev_sem(self, output_voxels, target_voxels):
        B, pC, pH, pW = output_voxels.shape
        tB, tH, tW = target_voxels.shape
        
        output_voxels = F.interpolate(output_voxels, size=(tH, tW), mode='bilinear', align_corners=False)

        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(target_voxels[0].detach().cpu().numpy())
        # plt.show()

        loss_bev_sem = CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels))
        return loss_bev_sem

    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).to(label)
        ones = ones.index_select(0, label.long())
        size.append(N)
        ones = ones.view(*size)
        ones = ones.transpose(2, 3)
        ones = ones.transpose(1, 2)
        return ones

    def loss_san_saw(self, san_saw_output, occ_gts):
        """
        san_saw_output = {
                'sem_pred':  inter_num,B*L,cls,H,W
                'embed_ori': inter_num,B*L,C,H,W
                'embed_san': inter_num,B*L,C,H,W
            }
        """
        occ_gts = occ_gts[0][self.history_queue_length:]  # B*L,H,W,D
        bev_sem_gts, _ = torch.max(occ_gts, dim=-1) # B*L, H,W
        gt_one_hot = self.get_one_hot(bev_sem_gts, self.num_classes)    # B*L,cls,H,W

        loss_dict = {}
        inter_num = san_saw_output['sem_pred'].shape[0]
        for index in range(inter_num):
            sem_pred = san_saw_output['sem_pred'][index]    # B*L,cls,H,W
            embed_ori = san_saw_output['embed_ori'][index]  # B*L,C,H,W
            embed_san = san_saw_output['embed_san'][index]  # B*L,C,H,W

            # sem_loss
            loss_bev_sem = self.loss_bev_sem(sem_pred, bev_sem_gts)
            # san_loss
            with torch.no_grad():
                embed_gt = []
                for j in self.selecte_classes:
                    mask = torch.unsqueeze(gt_one_hot[:, j, :, :], 1)
                    mask = F.interpolate(mask, size=embed_ori.size()[2:], mode='nearest')
                    out = (embed_ori * mask).float()
                    out = self.san_stage.IN(out)
                    embed_gt.append(out)
                embed_gt = sum(embed_gt)
                embed_gt = self.san_stage.relu(embed_gt)
            loss_san = 0.5 * F.smooth_l1_loss(embed_san, embed_gt)

            loss_dict.update({
                'loss_bev_sem_{}'.format(index): loss_bev_sem,
                'loss_san_{}'.format(index): loss_san,
            })
        return loss_dict
