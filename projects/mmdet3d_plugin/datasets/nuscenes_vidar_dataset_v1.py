#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy
import torch
import numpy as np

from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.parallel import DataContainer as DC

from .nuscenes_vidar_dataset_template import NuScenesViDARDatasetTemplate


@DATASETS.register_module()
class NuScenesViDARDatasetV1(NuScenesViDARDatasetTemplate):
    r"""nuScenes visual point cloud forecasting dataset.
    """
    def _mask_points(self, pts_list):
        assert self.ego_mask is not None
        # remove points belonging to ego vehicle.
        masked_pts_list = []
        for pts in pts_list:
            ego_mask = np.logical_and(
                np.logical_and(self.ego_mask[0] <= pts[:, 0],
                               self.ego_mask[2] >= pts[:, 0]),
                np.logical_and(self.ego_mask[1] <= pts[:, 1],
                               self.ego_mask[3] >= pts[:, 1]),
            )
            pts = pts[np.logical_not(ego_mask)]
            masked_pts_list.append(pts)
        pts_list = masked_pts_list
        return pts_list

    def union2one(self, previous_queue, future_queue):
        """
            previous_queue = history_len*['img', 'points', 'aug_param']   包含当前帧 -4 -3 -2 -1 0
            future_queue = future_len*['img', 'points', 'aug_param']   包含当前帧 0 1 2
        """
        # 1. get transformation from all frames to current (reference) frame
        ref_meta = previous_queue[-1]['img_metas'].data
        valid_scene_token = ref_meta['scene_token']
        # compute reference e2g_transform and g2e_transform.
        ref_e2g_translation = ref_meta['ego2global_translation']
        ref_e2g_rotation = ref_meta['ego2global_rotation']
        ref_e2g_transform = transform_matrix(
            ref_e2g_translation, Quaternion(ref_e2g_rotation), inverse=False)
        ref_g2e_transform = transform_matrix(
            ref_e2g_translation, Quaternion(ref_e2g_rotation), inverse=True)
        # compute reference l2e_transform and e2l_transform
        ref_l2e_translation = ref_meta['lidar2ego_translation']
        ref_l2e_rotation = ref_meta['lidar2ego_rotation']
        ref_l2e_transform = transform_matrix(
            ref_l2e_translation, Quaternion(ref_l2e_rotation), inverse=False)
        ref_e2l_transform = transform_matrix(
            ref_l2e_translation, Quaternion(ref_l2e_rotation), inverse=True)

        queue = previous_queue[:] + future_queue[1:]  # total_len: 4(history)+1(current)+2(future)
        # pts_list = [each['points'].data for each in queue]
        # if self.ego_mask is not None:
        #     pts_list = self._mask_points(pts_list)
        total_cur2ref_lidar_transform = []  # total_len*[4*4]  i帧到当前帧
        total_ref2cur_lidar_transform = []  # total_len*[4*4]  当前帧到i帧
        # total_pts_list = [] # total_len*[Np,5]  xyzit t是0~6
        for i, each in enumerate(queue):
            meta = each['img_metas'].data

            # # store points in the current frame.
            # cur_pts = pts_list[i].cpu().numpy().copy()
            # cur_pts[:, -1] = i
            # total_pts_list.append(cur_pts)

            # store the transformation from current frame to reference frame.
            curr_e2g_translation = meta['ego2global_translation']
            curr_e2g_rotation = meta['ego2global_rotation']
            curr_e2g_transform = transform_matrix(
                curr_e2g_translation, Quaternion(curr_e2g_rotation), inverse=False)
            curr_g2e_transform = transform_matrix(
                curr_e2g_translation, Quaternion(curr_e2g_rotation), inverse=True)

            curr_l2e_translation = meta['lidar2ego_translation']
            curr_l2e_rotation = meta['lidar2ego_rotation']
            curr_l2e_transform = transform_matrix(
                curr_l2e_translation, Quaternion(curr_l2e_rotation), inverse=False)
            curr_e2l_transform = transform_matrix(
                curr_l2e_translation, Quaternion(curr_l2e_rotation), inverse=True)

            # compute future to reference matrix.
            cur_lidar_to_ref_lidar = (curr_l2e_transform.T @
                                      curr_e2g_transform.T @
                                      ref_g2e_transform.T @
                                      ref_e2l_transform.T)
            total_cur2ref_lidar_transform.append(cur_lidar_to_ref_lidar)

            # compute reference to future matrix.
            ref_lidar_to_cur_lidar = (ref_l2e_transform.T @
                                      ref_e2g_transform.T @
                                      curr_g2e_transform.T @
                                      curr_e2l_transform.T)
            total_ref2cur_lidar_transform.append(ref_lidar_to_cur_lidar)

        # 2. Parse previous and future can_bus information.
        imgs_list = [each['img'].data for each in previous_queue]   # history_len*[imgs]
        metas_map = {}  # { {'can_bus': xyz+angle 初始帧为0,其余帧是delta_xyzangle,  'ref_lidar_to_cur_lidar': 4*4,当前帧到第i帧} }
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        ref_meta = previous_queue[-1]['img_metas'].data

        # occ flow
        occ_label_flag_list = [each['occ_label_flag'] for each in queue]
        if self.mem_efficient:  # mem-efficient下只加载ref帧的occ_label
            if self.use_fine_occ:
                segmentation_list = queue[self.queue_length]['gt_occ'].data     # history+ref+future, [H,W,D]
            else:
                segmentation_list = queue[self.queue_length]['segmentation'].data   # history+ref+future, [H,W,D]
                instance_list = queue[self.queue_length]['instance'].data if 'instance' in queue[self.queue_length].keys() else None
                flow_list = queue[self.queue_length]['flow'].data if self.turn_on_flow else None
        else:                   # 非mem-efficient下加载所有帧的occ_label
            segmentation_list = [each['segmentation'].data for each in queue]   # N_in_prev+N_in_out*[N_prev+N_fur, H,W,D]
            flow_list = [each['flow'].data for each in queue] if self.turn_on_flow else None
        # vel_steering
        vel_steering_list = np.array([each['vel_steering'] for each in future_queue])     # history+ref+future, 4

        # 2.2. Previous
        ref_can_bus = None
        history_can_bus = []             # history_len*[xyz+angle 初始帧为0,其余帧是delta_xyzangle]
        history2ref_lidar_transform = [] # history_len*[4*4] 第i帧到当前帧
        ref2history_lidar_transform = [] # history_len*[4*4] 当前帧到第i帧
        for i, each in enumerate(previous_queue):
            metas_map[i] = each['img_metas'].data

            if 'aug_param' in each:
                metas_map[i]['aug_param'] = each['aug_param']
            
            # store the transformation:
            history2ref_lidar_transform.append(
                total_cur2ref_lidar_transform[i]
            )  # current -> reference.
            ref2history_lidar_transform.append(
                total_ref2cur_lidar_transform[i]
            )  # reference -> current.

            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # Set the original point of this motion.
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus[:3] = 0
                new_can_bus[-1] = 0
                metas_map[i]['can_bus'] = new_can_bus
                history_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # Compute the later waypoint.
                # To align the shift and rotate difference due to the BEV.
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus[:3] = tmp_pos - prev_pos
                new_can_bus[-1] = tmp_angle - prev_angle
                metas_map[i]['can_bus'] = new_can_bus
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
                # compute history can_bus
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus_pos = np.array([0, 0, 0, 1]).reshape(1, 4)
                ref2prev_lidar_transform = ref2history_lidar_transform[-2]
                cur2ref_lidar_transform = history2ref_lidar_transform[-1]
                new_can_bus_pos = new_can_bus_pos @ cur2ref_lidar_transform @ ref2prev_lidar_transform
                new_can_bus_angle = new_can_bus[-1] - ref_can_bus[-1]
                new_can_bus[:3] = new_can_bus_pos[:, :3]
                new_can_bus[-1] = new_can_bus_angle
                history_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(metas_map[i]['can_bus'])

            # compute cur_lidar_to_ref_lidar transformation matrix for quickly align generated
            #  bev features to the reference frame.
            metas_map[i]['ref_lidar_to_cur_lidar'] = total_ref2cur_lidar_transform[i]
        # history can_bus store in first frame
        metas_map[0]['history_can_bus'] = history_can_bus

        # 2.3. Future
        current_scene_token = ref_meta['scene_token']
        ref_can_bus = None
        future_can_bus = []             # future_len*[xyz+angle 当前帧为0,其余帧是delta_xyzangle]
        future2ref_lidar_transform = [] # future_len*[4*4] 第i帧到当前帧
        ref2future_lidar_transform = [] # future_len*[4*4] 当前帧到第i帧
        for i, each in enumerate(future_queue):
            future_meta = each['img_metas'].data
            if future_meta['scene_token'] != current_scene_token:
                break

            # store the transformation:
            future2ref_lidar_transform.append(
                total_cur2ref_lidar_transform[i + len(previous_queue) - 1]
            )  # current -> reference.
            ref2future_lidar_transform.append(
                total_ref2cur_lidar_transform[i + len(previous_queue) - 1]
            )  # reference -> current.

            # can_bus information.
            if i == 0:
                new_can_bus = copy.deepcopy(future_meta['can_bus'])
                new_can_bus[:3] = 0
                new_can_bus[-1] = 0
                future_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(future_meta['can_bus'])
            else:
                new_can_bus = copy.deepcopy(future_meta['can_bus'])

                new_can_bus_pos = np.array([0, 0, 0, 1]).reshape(1, 4)
                ref2prev_lidar_transform = ref2future_lidar_transform[-2]
                cur2ref_lidar_transform = future2ref_lidar_transform[-1]
                new_can_bus_pos = new_can_bus_pos @ cur2ref_lidar_transform @ ref2prev_lidar_transform

                new_can_bus_angle = new_can_bus[-1] - ref_can_bus[-1]
                new_can_bus[:3] = new_can_bus_pos[:, :3]
                new_can_bus[-1] = new_can_bus_angle
                future_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(future_meta['can_bus'])

        ret_queue = previous_queue[-1]
        ret_queue['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        ret_queue.pop('aug_param', None)
        ret_queue.pop('gt_occ', None)

        metas_map[len(previous_queue) - 1]['future_can_bus'] = np.array(future_can_bus)
        metas_map[len(previous_queue) - 1]['future2ref_lidar_transform'] = (
            np.array(future2ref_lidar_transform))
        metas_map[len(previous_queue) - 1]['ref2future_lidar_transform'] = (
            np.array(ref2future_lidar_transform))
        metas_map[len(previous_queue) - 1]['total_cur2ref_lidar_transform'] = (
            np.array(total_cur2ref_lidar_transform))
        metas_map[len(previous_queue) - 1]['total_ref2cur_lidar_transform'] = (
            np.array(total_ref2cur_lidar_transform))

        ret_queue['img_metas'] = DC(metas_map, cpu_only=True)
        # ret_queue.pop('points')
        # ret_queue['gt_points'] = DC(
        #     torch.from_numpy(np.concatenate(total_pts_list, 0)), cpu_only=False)

        # occ flow
        ret_queue['occ_label_flag'] = occ_label_flag_list
        ret_queue['segmentation'] = segmentation_list
        if 'instance' in queue[self.queue_length].keys():
            ret_queue['instance'] = instance_list
        if self.turn_on_flow:
            ret_queue['flow'] = flow_list

        # vel_steering
        # ret_queue['vel_steering'] = vel_steering_list # NOTE: vel under global_coord, not use!
        # use vel under lidar_coord
        future_ego_pos = previous_queue[-1]['sdc_planning']   # ref+future, 3   x,y,yaw   cumsum
        future_ego_pos = np.concatenate([np.array([0.,0.,0.])[None], future_ego_pos], axis=0)
        future_ego_pos[:, 2] = future_ego_pos[:, 2] / 180 * np.pi
        # caculate vel
        ego_vw = (future_ego_pos[1:, -1] - future_ego_pos[:-1, -1]) / 0.5
        ego_v = np.linalg.norm(future_ego_pos[1:, :2] - future_ego_pos[:-1, :2], axis=-1) / 0.5
        ego_yaw = future_ego_pos[1:, -1] + np.pi/2
        ego_vx, ego_vy = ego_v * np.cos(ego_yaw), ego_v * np.sin(ego_yaw)
        # vel_steering
        vel = np.concatenate([ego_vx[:,None], ego_vy[:,None], ego_vw[:,None]], axis=-1)
        steering = vel_steering_list[:, -1][:,None]
        ret_queue['vel_steering'] = np.concatenate([vel, steering], axis=-1)

        # ego trajectory
        if 'rel_poses' in queue[0].keys():
            ## ref+future在ref_lidar坐标系下
            rel_poses = previous_queue[-1]['rel_poses'] # ref+future,2
            gt_modes = previous_queue[-1]['gt_modes']   # ref+future,3
            ## history+ref+future在prev_lidar坐标系下
            # rel_poses = np.array([each['rel_poses'] for each in queue]) # history+ref+future, 2
            # gt_modes = np.array([each['gt_modes'] for each in queue])     # history+ref+future, 3
            ret_queue['rel_poses'] = rel_poses
            ret_queue['gt_modes'] = gt_modes

        # sdc trajectory
        if 'sdc_planning' in queue[0].keys():
            sdc_planning = future_ego_pos[1:] - future_ego_pos[:-1]
            sdc_planning_mask = previous_queue[-1]['sdc_planning_mask']
            ## history+ref+future在prev_lidar坐标系下
            # sdc_planning = np.array([each['sdc_planning'][0] for each in queue])   # history+ref+future, 3  x,y,yaw
            # sdc_planning[:,:2] = rel_poses  # NOTE: rel_poses是delta_lidar   sdc_planning是delta_egocar_center  因为bev_feat是以lidar为原点，所以采用delta_lidar  实际差别只有1cm左右
            # sdc_planning[:, 2] = sdc_planning[:, 2] / 180 * np.pi   # 角度 -> 弧度
            # sdc_planning_mask = np.array([each['sdc_planning_mask'][0] for each in queue])   # history+ref+future, 2  valid=1
            ret_queue['sdc_planning'] = sdc_planning
            ret_queue['sdc_planning_mask'] = sdc_planning_mask
        if 'command' in queue[0].keys():
            ## ref+future在ref_lidar坐标系下，last_step的command
            command = previous_queue[-1]['command']
            ## history+ref+future在prev_lidar坐标系下，只有Go_Straight
            # command = np.array([each['command'] for each in queue]) # history+ref+future
            # command = np.array([np.where(mode==1)[0].item() for mode in gt_modes])
            ret_queue['command'] = command

        if 'sample_traj' in queue[0].keys():
            sample_traj = previous_queue[-1]['sample_traj'] # sample_num, Lout, 3
            ret_queue['sample_traj'] = sample_traj

        # gt_future_boxes   segmentation_bev
        if 'gt_future_boxes' in previous_queue[-1].keys():
            ## ref+future在ref_lidar坐标系下
            gt_future_boxes = previous_queue[-1]['gt_future_boxes'] # Lout, [N_box]  每帧的下一帧的gt_boxes,ref_lidar坐标系下
            ret_queue['gt_future_boxes'] = DC(gt_future_boxes, cpu_only=True)
        if 'segmentation_bev' in previous_queue[-1].keys():
            segmentation_bev = np.array(previous_queue[-1]['segmentation_bev']) # Lout,h,w  每帧的下一阵gt_boxes的segmentation_bev
            ret_queue['segmentation_bev'] = segmentation_bev

        if len(future_can_bus) < 1 + self.future_length:
            return None
        return ret_queue
        """
            ret_queue = {
                'img': Lin,Ncams,3,H,W                                              history+ref

                'img_metas': Lin_i = {                                              history+ref
                        'can_bus': xyz+angle 初始帧为0,其余帧是delta_xyzangle,  
                        'ref_lidar_to_cur_lidar': 4*4,当前帧(ref)到第i帧

                        只有第一帧(first)有：
                        'history_can_bus': history_len*(xyz+angle) 第一帧为0,其余朕是delta_xyzangle   history+ref

                        只有当前帧(ref)有：
                        'future_can_bus': future_len*(xyz+angle) 当前帧为0,其余帧是delta_xyzangle     ref+future
                        'future2ref_lidar_transform': future_len*4*4 第i帧到当前帧
                        'ref2future_lidar_transform': future_len*4*4 当前帧到第i帧
                        'total_cur2ref_lidar_transform': total_len*4*4  第i帧到当前帧
                        'total_ref2cur_lidar_transform': total_len*4*4  当前帧到第i帧
                }

                'gt_points': total_len*Np,5  xyzit t是0~6                           history+ref+future

                'segmentation':  total_len*[prev_pred+1+next_pred, H,W,D]            mem-efficient下,total_len=1只有ref帧; 非mem-efficient下, total_len=history+ref+future
                'flow':          total_len*[prev_pred+1+next_pred, 3,H,W,D]          mem-efficient下,total_len=1只有ref帧; 非mem-efficient下, total_len=history+ref+future
                'occ_label_flag':[total_len]  每帧是否在usable_index中,不在(前后有不同scene的)不求loss
                'rel_poses':     total_len,2                                        history+ref+future 从第i时刻到i+1时刻的(delta_x,delta_y) 即gt_planning
                'gt_modes':       total_len,3                                        history+ref+future [1 0 0]:Right  [0 1 0]:Left  [0 0 1]:Stright
            }
        """
