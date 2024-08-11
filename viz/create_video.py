import cv2
import os
import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes


def save_video_imgs_occ(images, occs, folder_path, fps=10):
    # video parameter
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(folder_path, 'camera_occ.mp4'), fourcc, fps, (width * 2, height))

    # wirte video
    for img_path, occ_path in zip(images, occs):
        img = cv2.imread(img_path)
        occ = cv2.imread(occ_path)

        if img.shape[0] != height or occ.shape[0] != height:
            occ = cv2.resize(occ, (width, height))

        # hstack img and occ
        combined_image = np.hstack((img, occ))

        video.write(combined_image)

    video.release()
    print(f"Video saved as {'camera_occ.mp4'} in {folder_path}")

def save_video_imgs(images, folder_path, output_video_name, fps=10):
    # video parameter
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(folder_path, output_video_name), fourcc, fps, (width, height))

    # wirte video
    for image_path in images:
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    print(f"Video saved as {output_video_name} in {folder_path}")

def create_video_from_images(folder_path, nusc, fps=10):
    images = []
    occs = []
    
    # subfolders
    subfolders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

    # timestamp
    timestamps = []
    for lidar_token in subfolders:
        lidar_data = nusc.get('sample_data', lidar_token)
        timestamps.append(lidar_data['timestamp'])

    # sort according timestamp
    combined = list(zip(subfolders, timestamps))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    subfolders, timestamps = zip(*sorted_combined)
    subfolders = list(subfolders)
    timestamps = list(timestamps)

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        image_path = os.path.join(subfolder_path, "img1.png")
        occ_path = os.path.join(subfolder_path, "occ.png")
        
        if os.path.exists(image_path):
            images.append(image_path)
        if os.path.exists(occ_path):
            occs.append(occ_path)

    if not images:
        print(f"No img1.png found in {folder_path}")
        return
    if not occs:
        print(f"No occ.png found in {folder_path}")
        return

    # video with front_camera and occs
    save_video_imgs_occ(images, occs, folder_path, fps=fps)

    # video with front_camera
    save_video_imgs(images, folder_path, output_video_name='camera.mp4', fps=fps)
    # video with occs
    save_video_imgs(occs, folder_path, output_video_name='occ.mp4', fps=fps)


def process_folders(base_folder, nusc, fps=10):
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            create_video_from_images(folder_path, nusc, fps)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        '--figs_path',
        type=str,
        default='./viz/figs',
        help='Path to visulized figs')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    base_folder_path = args.figs_path
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.root_path, verbose=True)
    # create video
    process_folders(base_folder_path, nusc, fps=args.fps)

if __name__ == '__main__':
    main()