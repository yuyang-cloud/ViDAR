'''
calculate planner metric same as stp3
'''
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt

from skimage.draw import polygon
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R

ego_width, ego_length = 1.85, 4.084

class PlanningMetric():
    def __init__(self):
        super().__init__()
        
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length

        self.category_index = {
            'human':[2,3,4,5,6,7,8],
            'vehicle':[14,15,16,17,18,19,20,21,22,23]
        }

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension
    
    def get_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        # import pdb; pdb.set_trace()
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(gt_agent_boxes,gt_agent_feats)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        return segmentation, pedestrian
    
    def get_birds_eye_view_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        '''
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        ego_lcf_feats: (B, 9) 
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        '''
        T = 6
        segmentation = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.tensor.cpu().numpy()  #(N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., :T*2].reshape(-1, 6, 2)    # B,Lout,2
        gt_agent_fut_mask = gt_agent_feats[..., T*2:T*3].reshape(-1, 6)     # B,Lout
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
        gt_agent_fut_yaw = gt_agent_feats[..., T*3+10:T*4+10].reshape(-1, 6, 1) # B,Lout,1
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:,6:7] = -1*(gt_agent_boxes[:,6:7] + np.pi/2) # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]
        
        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[0,i][27])
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a,y_a,yaw_a,agent_length, agent_width]
                    if (category_index in self.category_index['vehicle']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if (category_index in self.category_index['human']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)
        

        return segmentation, pedestrian
    
    def _get_poly_region_in_image(self,param):
        lidar2cv_rot = np.array([[1,0], [0,-1]])
        x_a,y_a,yaw_a,agent_length, agent_width = param
        trans_a = np.array([[x_a,y_a]]).T
        rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
                                [np.sin(yaw_a), np.cos(yaw_a)]])
        agent_corner = np.array([
            [agent_length/2, -agent_length/2, -agent_length/2, agent_length/2],
            [agent_width/2, agent_width/2, -agent_width/2, -agent_width/2]]) #(2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a #(2,4)
        # convert to cv frame
        agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar) \
            - self.bev_start_position[:2,None] + self.bev_resolution[:2,None] / 2.0).T / self.bev_resolution[:2] #(4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2


    def evaluate_single_coll(self, traj, segmentation, input_gt):
        '''
        traj: torch.Tensor (n_future, 2)
            自车lidar系为轨迹参考系
                ^ y
                |
                | 
                0------->
                        x
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())   # 4,2  ego在第i帧lidar为原点的bev下的坐标
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])    # NOTE:BUG? 32  rr:H_bev   cc:W_bev
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)  # 32,2  32*[H_bev, W_bev]  ego初始位置

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        # 轨迹坐标系转换为:
        #  ^ x
        #  |
        #  | 
        #  0-------> y
        trajs_ = copy.deepcopy(trajs)   # Lout,1,2  [X_pc,Y_pc] 第i帧lidar坐标系下的ego-delta
        trajs_[:,:,[0,1]] = trajs_[:,:,[1,0]] # can also change original tensor         Lout,1,2  [Y_pc,X_pc]  第i帧lidar坐标系下的ego-delta
        trajs_ = trajs_ / self.dx.to(trajs.device)  # Lout,1,2    [H_bev,W_bev]  第i帧lidar为原点的 bev坐标系下的ego-delta
        trajs_ = trajs_.cpu().numpy() + rc # Lout,32,2    [H_bev+delta, W_bev+delta] 第i帧lidar为原点 bev坐标系下 ego的future的位置

        r = trajs_[:,:,0].astype(np.int32)                              # Lout,32   bev坐标系下 ego的future_H  对应lidar的Y
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)                              # Lout,32   bev坐标系下 ego的future_W  对应lidar的X
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]   # bev坐标系下 ego的future_H
            cc = c[t]   # bev坐标系下 ego的future_W
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())  # ego的future_HW是否与segmentation有交集    segmentation是 第i帧lidar为原点的bev坐标系
        
        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(
            self, 
            trajs, 
            gt_trajs, 
            segmentation
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
            自车lidar系为轨迹参考系
            ^ y
            |
            | 
            0------->
                    x
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        '''
        B, n_future, _ = trajs.shape
        # import pdb; pdb.set_trace()
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)    # ego中心点与segmentation撞击次数
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)# ego-car与segmentation撞击次数

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], input_gt=True)    # Lout 每帧是否撞击

            xx, yy = trajs[i,:,0], trajs[i, :, 1]   # Lout 每一帧为相对第i帧lidar坐标系
            # lidar系下的轨迹转换到图片坐标系下
            yi = ((yy - self.bx[0]) / self.dx[0]).long()    # Lout 每一帧为相对第i帧lidar为原点的bev坐标系下 ego所在的H
            xi = ((xx - self.bx[1]) / self.dx[1]).long()    # Lout 每一帧为相对第i帧lidar为原点的bev坐标系下 ego所在的W

            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future) # Lout
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()  # segmentation为每一帧相对第i帧lidar为原点的bev坐标系

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], input_gt=False)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum.detach().cpu().numpy(), obj_box_coll_sum.detach().cpu().numpy()

    def compute_L2(self, trajs, gt_trajs, mask):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        mask: torch.Tensor (n_future)
        '''
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        # import pdb; pdb.set_trace()

        pred_len = len(mask)
        valid_len = mask.sum()
        mask = mask.float()

        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                ) * mask[i]
                for i in range(pred_len)
            )
            / valid_len
        )
        
        return ade
