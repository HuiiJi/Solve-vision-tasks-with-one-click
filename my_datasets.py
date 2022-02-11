# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)




class LoadImages:  # for inference
    def __init__(self, path):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        files = [p]
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        if any(videos):
            self.mode = 'video'
        if any(images):
            self.mode = 'image'
            
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv

        # if any(videos):
        #     self.new_video(videos[0])  # new video
        # else:
        #     self.cap = None
        # assert self.nf > 0, f'No images or videos found in {p}. ' \
        #                     f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.mode == 'video':
            self.cap = cv2.VideoCapture(path)
            img0 = None
            self.count += 1
        else:
            self.cap = None
            self.count += 1
            img0 = cv2.imread (path)  # BGR
        return path, img0, self.cap, self.mode

    def __len__(self):
        return self.nf  # number of files
