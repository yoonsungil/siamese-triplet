import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import pkg_resources
import os
import torch.distributed as dist


register_coco_instances("deepfashion2_train", {}, "coco_format/instance_train.json", "train/image")
register_coco_instances("deepfashion2_val", {}, "coco_format/instance_val.json", "val/image/")

config_path = "mask_rcnn_R_101_FPN_3x.yaml"
cfg_file = pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", config_path))

cfg = get_cfg()
cfg.merge_from_file(cfg_file)
cfg.DATASETS.TRAIN = ("deepfashion2_train",)
cfg.DATASETS.TEST = ("deepfashion2_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = '../output/model.pth'
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)#'content/drive/MyDrive/cocoeval'
trainer.train()