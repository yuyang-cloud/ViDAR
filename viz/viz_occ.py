import os

import mmcv
import open3d as o3d
import numpy as np
import torch
import pickle
from tqdm import tqdm
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 0
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

MO_CLS = [2,3,4,5,6,7,9,10]
VOXEL_SIZE =  [0.2, 0.2, 0.2]
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],  # 4 car  Crimson
        [233, 150, 70, 255],   # 5 cons. Veh  Orangered
        [255, 61, 99, 255],  # 6 motorcycle  Darkorange
        [0, 0, 230, 255], # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone  Red
        [255, 140, 0, 255],# 9 trailer  Slategrey
        [255, 99, 71, 255],# 10 truck Burlywood
        [0, 207, 191, 255],    # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegetation

        # movable objects time_change
        [255, 70, 255, 255],
        [255, 110, 255, 255],
        [255, 150, 255, 255],
        [255, 190, 255, 255],
        [255, 250, 250, 255],
], dtype=np.float32)



def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    points = torch.cat((occ_show[:, 0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occ_show[:, 1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occ_show[:, 2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[:, -1]


def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

def create_line_mesh(traj, radius=0.05, color=[1, 0, 0]):
    line_mesh = o3d.geometry.TriangleMesh()
    for i in range(len(traj) - 1):
        p1 = traj[i]
        p2 = traj[i + 1]
        
        # Compute the direction and length of the line segment
        direction = p2 - p1
        length = np.linalg.norm(direction)
        direction = direction / length

        # Create a cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
        cylinder.paint_uniform_color(color)

        # Compute the rotation to align the cylinder
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation_matrix = np.eye(3)
        else:
            axis = np.cross(z_axis, direction)
            angle = np.arccos(np.dot(z_axis, direction))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        
        # Apply the rotation and translation
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        cylinder.translate(p2 - direction * length / 2)

        line_mesh += cylinder
    return line_mesh

def create_arrow(traj, radius=0.05, color=[1, 0, 0]):
    p1 = traj[-2]
    p2 = traj[-1]

    # Compute the direction and length of the last segment
    direction = p2 - p1
    length = np.linalg.norm(direction)
    direction = direction / length

    # Create an arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=radius * 3, cone_height=radius * 6,
        cylinder_radius=radius, cylinder_height=length
    )
    arrow.paint_uniform_color(color)

    # Compute the rotation to align the arrow
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        rotation_matrix = np.eye(3)
    else:
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # Apply the rotation and translation
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    
    # Move the arrow to the end of the last segment, adjusting to make sure it connects properly
    arrow.translate(p2 - direction * length / 2)
    
    return arrow


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4, traj=None,):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    ego_pcd = o3d.geometry.PointCloud()
    ego_points = generate_the_ego_car()
    ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    ego_pcd_colors = np.ones_like(ego_points) * [0., 0.5, 1.0]
    ego_pcd.colors = o3d.utility.Vector3dVector(ego_pcd_colors)
    vis.add_geometry(ego_pcd)

    if traj is not None:
        traj_height = np.ones(traj.shape[0]) * ego_points[:, -1].max()
        traj = np.concatenate([traj, traj_height[:, None]], axis=-1)
        traj[:, [0, 1]] = traj[:, [1, 0]]
        traj[:, 0] += ego_points[:, 0].max()
        traj = np.concatenate([np.array([ego_points[:,0].max(), 0., ego_points[:,-1].max()])[None,:], traj], axis=0)
        # # double y_axit to visulize more clear
        # traj_diff = traj[1:] - traj[:-1]
        # traj[1:] += traj_diff
        # radius
        radius = 0.08
        color=[1, 0, 0]
        # Create line mesh
        line_mesh = create_line_mesh(traj, radius=radius, color=color)
        vis.add_geometry(line_mesh)
        # Create arrow
        arrow = create_arrow(traj, radius=radius, color=color)
        vis.add_geometry(arrow)

    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0], traj=None):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels = voxel2points(occ_state, occ_show, voxel_size)
    pcd[:, 1] = -pcd[:, 1]
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels.int()]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=True,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4,
        traj=traj,
    )
    return vis


def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0] // 2
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--future-length',
        type=int,
        default=1,
        help='Number of visulize future frames')
    parser.add_argument(
        '--show_mo_time_change',
        action='store_true',
        help='whether not to visulize movable objects with time_change colors')
    parser.add_argument(
        '--show_traj',
        action='store_true',
        help='whether not to visulize trajctories')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./viz/figs',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    # load predicted results
    results_dir = args.res

    # load dataset information
    info_path = \
        args.root_path + 'nuscenes/nuscenes_infos_temporal_%s_new.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # gt nuscenes-Occupancy
    nuscocc_path = args.root_path + "nuScenes-Occupancy/"
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')

    # create window once
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    view_control = vis.get_view_control()
    # look_at = np.array([-0.185, 0.513, 3.485])
    # front = np.array([-0.974, -0.055, 0.221])
    # up = np.array([0.221, 0.014, 0.975])
    # zoom = np.array([0.08])
    # view_control.set_lookat(look_at)
    # view_control.set_front(front)
    # view_control.set_up(up)
    # view_control.set_zoom(zoom)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.line_width = 80

    segmentation_files = os.listdir(results_dir)
    segmentation_files.sort(key=lambda x: (x.split("_")[1]))

    for cnt, file_ in enumerate(tqdm(segmentation_files)):
        scene_token = file_.split("_")[0]
        lidar_token = file_.split("_")[1][:-4]

        # load gt
        gt_file = nuscocc_path+"scene_"+scene_token+"/occupancy/"+lidar_token+".npy"
        gt_occ_semantic =  np.load(gt_file,allow_pickle=True)
        gt_occ_semantic = gt_occ_semantic[gt_occ_semantic[:, -1]!=0]
        gt_occ_semantic_refine = np.zeros_like(gt_occ_semantic)
        gt_occ_semantic_refine[:, 0] = gt_occ_semantic[:, 2]    # X
        gt_occ_semantic_refine[:, 1] = gt_occ_semantic[:, 1]    # Y
        gt_occ_semantic_refine[:, 2] = gt_occ_semantic[:, 0]    # Z
        gt_occ_semantic_refine[:, 3] = gt_occ_semantic[:, 3]    # label

        # load occ_pred
        pred_data = np.load(os.path.join(results_dir, file_), allow_pickle=True)
        pred_occ_semantic =  pred_data['occ_pred']
        pred_occ_semantic_to_draw=np.zeros((0,4))
        for t in range(0, args.future_length):
            pred_mo_cur = pred_occ_semantic[t]
            pred_mo_cur = np.array(pred_mo_cur)
            if args.show_mo_time_change:
                # static object -> frame 0
                if t == 0:
                    pred_so_cur = np.array([x for x in pred_mo_cur if x[-1] not in MO_CLS])
                    pred_occ_semantic_to_draw = np.concatenate((pred_occ_semantic_to_draw, pred_so_cur))
                # movable object
                pred_mo_cur = np.array([x for x in pred_mo_cur if x[-1] in MO_CLS])
                pred_mo_cur[:, -1] = int(t+17)
            pred_occ_semantic_to_draw = np.concatenate((pred_occ_semantic_to_draw, pred_mo_cur))
        pred_occ_semantic_to_draw[:, [0, 1]] = pred_occ_semantic_to_draw[:, [1, 0]] # y,x -> x,y

        # load pose_pred
        if args.show_traj:
            pred_traj = pred_data['pose_pred']
        else:
            pred_traj = None

        # load info
        for info_i in dataset['infos']:
            if info_i['scene_token'] == scene_token and info_i['lidar_token'] == lidar_token:
                cam_info_i = info_i['cams']
        
        # load imgs
        imgs = []
        for view in views:
            img = cv2.imread(cam_info_i[view]['data_path'])
            imgs.append(img)

        # Clear previous geometries and update with new data
        vis.clear_geometries()

        # occ_canvas
        voxel_size = VOXEL_SIZE
        pred_occ = pred_occ_semantic_to_draw
        voxel_show = pred_occ[pred_occ[:, -1] != FREE_LABEL][:, :3]
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0], traj=pred_traj)

        if args.draw_gt:
            voxel_label = gt_occ_semantic_refine
            voxel_show = voxel_label[voxel_label[:, -1] != FREE_LABEL][:, :3]
            vis = show_occ(torch.from_numpy(voxel_label), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                            offset=[0, voxel_label.shape[0] * voxel_size[0] * 1.2 * 1, 0])

        # 加载viewpoint.json
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters('./viz/viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        # # 保留viewpoint.json
        # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('./viz/viewpoint.json', param)
        # vis.destroy_window()
        # breakpoint()


        occ_canvas = vis.capture_screen_float_buffer(do_render=True)
        occ_canvas = np.asarray(occ_canvas)
        occ_canvas = (occ_canvas * 255).astype(np.uint8)
        occ_canvas = occ_canvas[..., [2, 1, 0]]
        occ_canvas_resize = cv2.resize(occ_canvas, (canva_size, canva_size), interpolation=cv2.INTER_CUBIC)

        big_img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                       dtype=np.uint8)
        big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        big_img[900 + canva_size * scale_factor:, :, :] = img_back
        big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3),
                                       int(900 / scale_factor * 2 + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        big_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
                w_begin:w_begin + canva_size, :] = occ_canvas_resize

        if args.format == 'image':
            out_dir = os.path.join(vis_dir, f'{scene_token}', f'{lidar_token}')
            mmcv.mkdir_or_exist(out_dir)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f'img{i}.png'), img)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
        elif args.format == 'video':
            cv2.putText(big_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{scene_token}', (5, 35), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{lidar_token[:5]}', (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == 'video':
        vout.release()
    vis.destroy_window()


if __name__ == '__main__':
    main()