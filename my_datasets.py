
from pathlib import Path
import cv2

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


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
            img0 = cv2.imread(path)  # BGR
            self.count += 1
        return path, img0, self.cap, self.mode

    def __len__(self):
        return self.nf  # number of files


