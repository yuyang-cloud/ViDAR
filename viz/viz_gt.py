# Developed by Junyi Ma
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

from tqdm import tqdm
import pickle
import numpy as np
from mayavi import mlab
from tqdm import trange
import os
from xvfbwrapper import Xvfb

os.environ['ETS_TOOLKIT'] = 'qt5'
# export QT_QPA_PLATFORM='offscreen' 
mlab.options.offscreen = True

def viz_occ(occ, occ_mo, file_name, voxel_size, show_so_sem_class, show_mo_time_change, vis_mo=False):

    vdisplay = Xvfb(width=1, height=1)
    vdisplay.start()

    mlab.figure(size=(800,800), bgcolor=(1,1,1))

    plt_plot_occ = mlab.points3d(
        occ[:, 0] * voxel_size,
        occ[:, 1] * voxel_size,
        occ[:, 2] * voxel_size,
        occ[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=0.9,
        vmin=1,
    )
    if show_so_sem_class:
        colors_occ = np.array(
            [
                [255, 120, 50, 255],
                [100, 230, 245, 255],
                [100, 80, 250, 255],
                [100, 150, 245, 255],
                [175, 0, 75, 255],
                [30, 60, 150, 255],
                [255, 30, 30, 255],
                [255, 40, 200, 255],
                [150, 30, 90, 255],
                [80, 30, 180, 255],
                [255, 0, 255, 255],
                [255, 240, 150, 255],
                [75, 0, 75, 255],
                [150, 240, 80, 255],
                [135, 60, 0, 255],
                [0, 175, 0, 255],
            ]
        ).astype(np.uint8) 
    else:
        colors_occ = np.array(
            [
                [152, 251, 152, 255],
                [152, 251, 152, 255],
                [152, 251, 152, 255],
                [152, 251, 152, 255],
                [152, 251, 152, 255],
            ]
        ).astype(np.uint8)    
    plt_plot_occ.glyph.scale_mode = "scale_by_vector"
    plt_plot_occ.module_manager.scalar_lut_manager.lut.table = colors_occ

    if vis_mo:
        plt_plot_mov = mlab.points3d(
            occ_mo[:, 0] * voxel_size,
            occ_mo[:, 1] * voxel_size,
            occ_mo[:, 2] * voxel_size,
            occ_mo[:, 3],
            colormap="viridis",
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=0.9,
            vmin=1,
        )
        if show_mo_time_change:
            colors_occ_mo = np.array(
                [
                    [255, 70, 255, 255],
                    [255, 110, 255, 255],
                    [255, 150, 255, 255],
                    [255, 190, 255, 255],
                    [255, 250, 250, 255],
                ]
            ).astype(np.uint8)
        else:
            colors_occ_mo = np.array(
                [
                    [100, 230, 245, 255],
                    [100, 80, 250, 255],
                    [100, 150, 245, 255],
                    [175, 0, 75, 255],
                    [30, 60, 150, 255],
                    [150, 30, 90, 255],
                    [80, 30, 180, 255],
                    [255, 30, 30, 255],
                ]
            ).astype(np.uint8)    
        plt_plot_mov.glyph.scale_mode = "scale_by_vector"
        plt_plot_mov.module_manager.scalar_lut_manager.lut.table = colors_occ_mo

    fig_dir = "./figs"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    mlab.savefig(os.path.join(fig_dir, file_name[:-4]+".png"))
    vdisplay.stop()


def main():

    show_so_sem_class = True    # Static Object show semantic_class
    show_mo_time_change = False # Moving Object show time_change or semantic_class
    vis_mo = False              # Moving Object visulization

    nuscocc_path = "../data/nuScenes-Occupancy/"
    cam4docc_path = "../data/cam4docc/MMO/segmentation/"

    segmentation_files = os.listdir(cam4docc_path)
    segmentation_files.sort(key=lambda x: (x.split("_")[1]))
    index = 0

    for file_ in tqdm(segmentation_files):

        scene_token = file_.split("_")[0]
        lidar_token = file_.split("_")[1]

        gt_file = nuscocc_path+"scene_"+scene_token+"/occupancy/"+lidar_token[:-4]+".npy"
        gt_occ_semantic =  np.load(gt_file,allow_pickle=True)
        gt_occ_semantic = gt_occ_semantic[gt_occ_semantic[:, -1]!=0]
        # gt_occ_semantic = gt_occ_semantic[::2]    # 下采样
        gt_occ_semantic_refine = np.zeros_like(gt_occ_semantic)
        gt_occ_semantic_refine[:, 0] = gt_occ_semantic[:, 2]
        gt_occ_semantic_refine[:, 1] = gt_occ_semantic[:, 1]
        gt_occ_semantic_refine[:, 2] = gt_occ_semantic[:, 0]
        if show_so_sem_class:
            gt_occ_semantic_refine[:, 3] = gt_occ_semantic[:, 3]
        else:
            gt_occ_semantic_refine[:, 3] = 1    # occupied

        gt_mo_semantic =  np.load(cam4docc_path+file_,allow_pickle=True)['arr_0']

        gt_mo_semantic_to_draw=np.zeros((0,4))
        for t in range(0,4):
            gt_mo_cur = gt_mo_semantic[t]
            gt_mo_cur = np.array(gt_mo_cur)
            # gt_mo_cur = gt_mo_cur[::2]    # 下采样
            if show_mo_time_change:
                gt_mo_cur[:, -1] = int(t+1)
            gt_mo_semantic_to_draw = np.concatenate((gt_mo_semantic_to_draw, gt_mo_cur))

        viz_occ(gt_occ_semantic_refine, gt_mo_semantic_to_draw, file_, voxel_size=0.2,
                show_so_sem_class=show_so_sem_class, show_mo_time_change=show_mo_time_change, vis_mo=vis_mo)

        index += 1


if __name__ == "__main__":
    main()