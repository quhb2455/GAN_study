import os
import mmcv

'''
- mmdetection/config/_base_ 에 있는 datasets와 models의 세팅을 여기서 변경
- mmdetection/config 에 각 모델별로 dataset과 model이 setting 되어 있음
- balloon dataset에 대해서는 안되어 있기 때문에 여기서 설정해줌
- coco format을 쓰기 때문에 _base_/datasets/coco_detection.py의 설정을 그대로 가져와서 사용
'''

_base_ = 'mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# data annotation에 맞게 head의 num_class 조절
model = dict(
    roi_head = dict(
        bbox_head = dict(num_classes=1),
        mask_head = dict(num_classes=1)
    )
)

# dataset setting 수정
dataset_type = 'CocoDataset'
classes = ('balloon', )
data = dict(
    train=dict(
        type=dataset_type,
        ann_file='../../data/balloon_dataset/balloon/train/coco_via_region_data.json',
        img_prefix='../../data/balloon_dataset/balloon/train/',
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file='../../data/balloon_dataset/balloon/val/coco_via_region_data.json',
        img_prefix='../../data/balloon_dataset/balloon/val/',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file='../../data/balloon_dataset/balloon/val/coco_via_region_data.json',
        img_prefix='../../data/balloon_dataset/balloon/val/',
        classes=classes
    )
)

# pretrained model 사용
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'