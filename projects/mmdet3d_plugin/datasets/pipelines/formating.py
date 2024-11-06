
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results



@PIPELINES.register_module()
class OccDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(OccDefaultFormatBundle3D, self).__call__(results)

        if 'gt_occ' in results.keys():
            results['gt_occ'] = DC(to_tensor(results['gt_occ']), stack=True)
        if 'gt_occ' in results.keys():
            results['segmentation'] = DC(to_tensor(results['segmentation']), stack=True)
        if 'gt_occ' in results.keys():
            results['instance'] = DC(to_tensor(results['instance']), stack=True)
        if 'gt_occ' in results.keys():
            results['attribute_label'] = DC(to_tensor(results['attribute_label']), stack=True)
        if 'gt_occ' in results.keys():
            results['flow'] = DC(to_tensor(results['flow']), stack=True)
        if 'gt_vel' in results.keys():
            results['gt_vel'] = DC(to_tensor(results['gt_vel']), stack=False)

        return results