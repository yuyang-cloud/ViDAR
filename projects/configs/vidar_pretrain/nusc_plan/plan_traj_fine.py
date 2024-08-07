_base_ = [
    '../../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
occ_path = "./data/nuScenes-Occupancy"
use_fine_occ = True

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 0.2]
occ_size = [512, 512, 40]

# plan_grid_conf
plan_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Dataloader.
queue_length = 4 # history frame num input  (不包含ref)
memory_queue_len = 1 # memory queue

future_queue_length_train = 6 # future frame num input
future_pred_frame_num_train = 6 # future frame num pred training
future_queue_length_test = 6   # future frame num input
future_pred_frame_num_test = 6 # future frame num pred testing

plan_decoder_layer_num = 1
future_decoder_layer_num = 3
frame_loss_weight = [
    [1],  # for frame 0.  ref_frame
    [1],  # for frame 1.  ref_frame+1
    [0]  # ignore.
]
only_generate_dataset = False # only generate cam4docc dataset, no forward
mem_efficient = True         # only load ref_frame's occ_data
supervise_all_future = True  # select which future to predict occ (defalut=True, when mem_efficient=True, all future generate occ) (when mem-efficient=False, 可以为False选取部分帧计算loss从而save gpu)
load_frame_interval = None  # use 1/8 nuscenes dataset for faster evaluation.

turn_on_flow = False         # turn_on_flow=True: load flow_label and predict flow througn flow_branch
turn_on_plan = True

# ViDAR model.
vidar_head_pred_history_frame_num = 0   # for aux loss  同时pred每帧的前history帧
vidar_head_pred_future_frame_num = 0    # for aux loss  同时pred每帧的后future帧
vidar_head_per_frame_loss_weight = (1.0, )

# Latent Rendering Structure.
future_latent_render_keep_idx = (),
latent_render_act_fn = 'sigmoid'
latent_render_layer_idx = ()
latent_render_grid_step = 0.5

ida_aug_conf = {
    "reisze": [720, 765, 810, 855, 900, 945, 990, 1035, 1080],  #  (0.8, 1.2)
    "crop": (0, 0, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
use_separate_classes = True
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction', 
    'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
    'truck', 'driveable_surface', 'other', 'sidewalk',
    'terrain', 'mannade', 'vegetation',
]
empty_idx = 0
num_cls = len(class_names) + 1

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
pred_height = 16

model = dict(
    type='ViDAR',
    turn_on_flow=turn_on_flow,
    turn_on_plan=turn_on_plan,
    memory_queue_len=memory_queue_len,
    use_grid_mask=True,
    video_test_mode=True,
    only_generate_dataset=only_generate_dataset,
    mem_efficient=mem_efficient,
    supervise_all_future=supervise_all_future,

    # BEV configuration.
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_,
    bev_w=bev_w_,

    # Predict frame num.
    future_pred_frame_num=future_pred_frame_num_train,
    test_future_frame_num=future_pred_frame_num_test,

    # visulization
    _viz_pcd_flag=False,
    _viz_pcd_path='work_dirs/plan_traj_fine/results',

    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    future_pred_head=dict(
        type='ViDARHeadV1',
        num_classes=num_cls,
        history_queue_length=queue_length,
        memory_queue_len=memory_queue_len,
        soft_weight=False,
        turn_on_flow=False, # Occ Head
        obj_motion_norm=False,
        pred_history_frame_num=vidar_head_pred_history_frame_num,
        pred_future_frame_num=vidar_head_pred_future_frame_num,
        per_frame_loss_weight=vidar_head_per_frame_loss_weight,

        ray_grid_num=512,
        ray_grid_step=1.0,

        use_ce_loss=True,
        use_dist_loss=False,
        use_dense_loss=True,

        num_pred_fcs=1,  # head for point cloud prediction.
        num_pred_height=pred_height,  # Predict BEV instead of 3D space occupancy.

        use_can_bus=False,    # use future gt traj
        use_plan_traj=True, # use future pred traj      4D-Occ-Pred: must use_plan_traj=True
        use_command=False,
        use_vel_steering=False,
        use_vel=False,
        use_steering=False,
        condition_ca_add='ca',
        can_bus_norm=True,
        can_bus_dims=(0, 1, 2, 17),
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        loss_weight=frame_loss_weight,
        loss_weight_cfg=dict(               # un-commented: multi-loss      commented: only CE-loss
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        prev_render_neck=dict(
            type='LatentRenderingV2',
            occ_flow='occ',
            embed_dims=_dim_, 
            occ_render=False,
            sem_render=False,
            sem_norm=False,
            sem_gt_train=False,
            ego_motion_ln=False,
            obj_motion_ln=False,
            pred_height=16, 
            num_cls=num_cls,
            num_pred_fcs=0,
            grid_step=latent_render_grid_step, 
            grid_num=256,
            reduction=16, 
            act=latent_render_act_fn, 
        ),
        transformer=dict(
            type='PredictionTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='PredictionDecoder',
                keep_idx=future_latent_render_keep_idx,
                num_layers=future_decoder_layer_num,
                return_intermediate=True,
                transformerlayers=dict(
                    type='PredictionTransformerLayer',
                    attn_cfgs=[
                        # layer-1: deformable self-attention.
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=memory_queue_len,
                        ),
                        # layer-2: deformable cross-attention,
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=memory_queue_len),
                        # layer-3: cross-attention with action condition,
                        dict(
                            type='GroupMultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'cross_attn_action', 'norm', 'ffn', 'norm'))
            )),
    ),
    plan_head=dict(
        type='PlanHead_v3',
        with_adapter=True,
        plan_grid_conf=plan_grid_conf,
        bev_h=bev_h_,
        bev_w=bev_w_,
        transformer=dict(
            type='PlanTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='PlanDecoder',
                num_layers=plan_decoder_layer_num,
                return_intermediate=False,
                transformerlayers=dict(
                    type='PlanTransformerLayer',
                    attn_cfgs=[
                        # layer-1: temporal self-attention.
                        dict(
                            type='GroupMultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,),
                        # layer-2: spatial cross-attention,
                        dict(
                            type='GroupMultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))
            )),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_planning=dict(type='PlanningLoss'),
        loss_collision=[dict(type='CollisionLoss', delta=0.0, weight=2.5),
                        dict(type='CollisionLoss', delta=0.5, weight=1.0),
                        dict(type='CollisionLoss', delta=1.0, weight=0.25)],
    ),
    pts_bbox_head=dict(
        type='ViDARBEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=num_cls,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            embed_dims=_dim_,
            encoder=dict(
                type='CustomBEVFormerEncoder',
                keep_idx=latent_render_layer_idx,
                latent_rendering_lid=latent_render_layer_idx,
                num_layers=6,   # 6
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayerV2',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            # !!!!!!! DECODER NOT USED !!!!!!!
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_cls),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'NuScenesViDARDatasetV1'
data_root = 'data/nuscenes/'
cam4docc_dataset_path = 'data/cam4docc/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),

    # new data augmentation.
    dict(type='CropResizeFlipImage', data_aug_conf=ida_aug_conf, training=True, debug=False),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),

    dict(type='LoadOccupancy', to_float32=True, occ_path=occ_path, grid_size=occ_size, unoccupied=empty_idx, pc_range=point_cloud_range, use_fine_occ=use_fine_occ, 
         time_history_field=queue_length, time_future_field=future_pred_frame_num_train, test_mode=False),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'aug_param', 'gt_occ', 'occ_label_flag', 'rel_poses', 'gt_modes', 'vel_steering',
                                       'sdc_planning', 'sdc_planning_mask', 'command', 'sample_traj', 'gt_future_boxes'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadOccupancy', to_float32=True, occ_path=occ_path, grid_size=occ_size, unoccupied=empty_idx, pc_range=point_cloud_range, use_fine_occ=use_fine_occ, 
         time_history_field=queue_length, time_future_field=future_pred_frame_num_train, test_mode=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ', 'occ_label_flag', 'rel_poses', 'gt_modes', 'vel_steering',
                                       'sdc_planning', 'sdc_planning_mask', 'command', 'sample_traj', 'segmentation_bev'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train_new.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        use_separate_classes=use_separate_classes,
        mem_efficient=mem_efficient,
        use_fine_occ=use_fine_occ,
        turn_on_flow=turn_on_flow,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        future_length=future_queue_length_train,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        load_frame_interval=load_frame_interval,
        plan_grid_conf=plan_grid_conf,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val_new.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             use_separate_classes=use_separate_classes,
             mem_efficient=mem_efficient,
             use_fine_occ=use_fine_occ,
             turn_on_flow=turn_on_flow,
             classes=class_names, modality=input_modality, samples_per_gpu=1,

             # some evaluation configuration.
             queue_length=queue_length,
             future_length=future_queue_length_test,
             ego_mask=(-0.8, -1.5, 0.8, 2.5),
             plan_grid_conf=plan_grid_conf,
             ),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val_new.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              use_separate_classes=use_separate_classes,
              mem_efficient=mem_efficient,
              use_fine_occ=use_fine_occ,
              turn_on_flow=turn_on_flow,
              classes=class_names, modality=input_modality,

              # some evaluation configuration.
              queue_length=queue_length,
              future_length=future_queue_length_test,
              ego_mask=(-0.8, -1.5, 0.8, 2.5),
              plan_grid_conf=plan_grid_conf,
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # 0.1 in pre-train.
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'work_dirs/action_codition_fine/epoch_24.pth' # 'pretrained/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='SetEpochInfoHook')]
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
