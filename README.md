# Requirements

The installation step is similar to [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md).
For convenience, we list the steps below:
```bash
conda create -n vidar python=3.8 -y
conda activate vidar

# torch must < 1.11.0 which is required by mmdet3d
# 经测试，高版本的CUDA向下兼容
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
conda install -c omgarcia gcc-6 # (optional) Make sure the GCC version compatible with CUDA
```

Install some other required packges and Detectron2.
```bash
pip install setuptools einops fvcore seaborn ninja iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 casadi==3.5.5 pytorch-lightning==1.2.5 lyft_dataset_sdk nuscenes-devkit plyfile networkx==2.2 trimesh==2.35.39 yapf==0.40.1

# Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# or install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```


Install MM-Series packages.
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# Install mmdetection3d from source codes.
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install 
# Warning: CUDA version has a minor version mismatch with the version that was used to compile PyTorch(11.1). Most likely this shouldn't be a problem. 不影响编译
```


# Setup
```bash
git clone https://github.com/yuyang-cloud/ViDAR

cd ViDAR
mkdir pretrained
cd pretrained & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth

# Install DCNv3
cd ViDAR
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
python setup.py install

# Install chamferdistance library.
cd ViDAR
cd third_lib/chamfer_dist/chamferdist/
pip install .
```


# Prepare Data

**Folder structure**
```
ViDAR
├── projects/
├── tools/
├── pretrained/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train_new.pkl
|   |   ├── nuscenes_infos_temporal_val_new.pkl
│   ├── cam4docc
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   ├── nuScenes-Occupancy/
```

**1. nuscene**
```bash
cd ViDAR
mkdir data
ln -s /path/to/nuscenes data/nuscenes
ln -s /path/to/nuscnens/can_bus data/can_bus
```

**2. nuscene.pkl**

1) 下载已经预处理好的 [nuscenes_infos_temporal_train_new.pkl](https://github.com/yuyang-cloud/ViDAR/releases/tag/nuscenes_infos_temporal_train_new.pkl) 和 [nuscenes_infos_temporal_val_new.pkl](https://github.com/yuyang-cloud/ViDAR/releases/tag/nuscenes_infos_temporal_val_new.pkl)
2) 将```.pkl```放入nuscenes文件夹

```bash
mv nuscenes_infos_temporal_train_new.pkl /path/to/nuscenes
mv nuscenes_infos_temporal_val_new.pkl /path/to/nuscenes
```

**3. Cam4DOcc**

代码在第一次运行过程中，会生成并保存Cam4DOcc中定义的Occupancy数据（Inflated 3D Bounding Box形式的Occ）

先创建空文件路径，后续程序运行的第一个epoch会在路径中生成数据：
```bash
mkdir /path/to/save_data/cam4docc   # 数据较大~650G,建议放在数据盘中
cd /path/to/save_data/cam4docc
mkdir GMO GMO_lyft MMO MMO_lyft
ln -s /path/to/save_data/cam4docc ViDAR/data
```

**4. nuScenes-Occupancy**

如已有这个数据集，直接软链接；如未下载此数据集，可以暂时先不下载，先跑Cam4DOcc的Occupancy数据

压缩包下载地址：
https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md

```bash
mv nuScenes-Occupancy-v0.1.7z /path/to/save_data
cd /path/to/save_data
7za x nuScenes-Occupancy-v0.1.7z
mv nuScenes-Occupancy-v0.1 nuScenes-Occupancy
ln -s /path/to/save_data/nuScenes-Occupancy ViDAR/data/
```




# Train and Evaluate

**1. Generate Cam4DOcc Dataset**

此配置文件只会生成cam4docc数据，没有模型forward过程，所以不会占用GPU显存

```bash
CONFIG=projects/configs/vidar_pretrain/nusc/cam4docc_generate_dataset.py
GPU_NUM=8   # 建议GPU_NUM给越大越好，因为不会占用GPU显存，多进程生成数据速度更快；8卡约1~2天

./tools/dist_train.sh ${CONFIG} ${GPU_NUM}  # 生成训练集数据

CKPT=work_dirs/cam4docc_generate_dataset/epoch_1.pth
./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}  # 生成测试集数据
```


**2. Train**

```bash
CONFIG=path/to/config.py
GPU_NUM=8

./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```
Configs:

1) ```projects\configs\vidar_pretrain\nusc\action_condition.py``` : use Cam4DOcc dataset, only predict instances(Things) with inflated bbox occupancy.
2) ```projects\configs\vidar_pretrain\nusc\action_condition_fine.py``` : use nuScenes-Occupancy dataset, predict both Things and Stuff with fine-grained occupancy.
3) ```projects\configs\vidar_pretrain\nusc\plan_traj_fine.py``` : use nuScenes-Occupancy dataset, plan trajectories based on fine-grained occupancy predictions.

Train with planning 包括两阶段：
```bash
GPU_NUM=8

# Step1. Train the model with action_condition (gt action):
CONFIG=projects/configs/vidar_pretrain/nusc/action_condition_fine.py
./tools/dist_train.sh ${CONFIG} ${GPU_NUM}

# Step2. Train the model with planning (pred action):
CONFIG=projects/configs/vidar_pretrain/nusc/plan_traj_fine.py
# 注意修改上述配置文件中 load_from='work_dirs/action_condition_fine/epoch24.pth'为Step1的预训练权重
./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```

**3. Evaluate**

```bash
CONFIG=path/to/vidar_config.py
CKPT=work_dirs/config_file_name/epoch_24.pth
GPU_NUM=8

./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}
```

**4. Visulization**

```bash
# Step1. 修改config中如下参数:
CONFIG=projects/configs/vidar_pretrain/nusc/plan_traj_fine.py
_viz_pcd_flag=True
_viz_pcd_path='work_dirs/config_file_name/results'

# Step2. Inference
CKPT=work_dirs/config_file_name/epoch_24.pth
GPU_NUM=8
./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}

# Step3. Visulization
python viz/viz_occ.py ${_viz_pcd_path} # 结果保存在 ./viz/figs/
# optional args:
--future_length 1 
--show_mo_time_change # whether not to visulize movable objects with time_change colors
--show_traj # whether not to visulize pred ego_trajs 推荐show_traj配合show_mo_time_change使用，并将future_length调到3及以上

# Step4. Create Video
python viz/create_video.py --figs_path ./viz/figs
```