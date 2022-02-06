import argparse
from pathlib import Path
import os
import cv2
from PIL import Image
import numpy as np
from DMSHNet import DMSHN
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from my_datasets import  LoadImages
from my_general import check_img_size, check_requirements, increment_path


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str,
                    default='images', help='source')
parser.add_argument('--img-size', type=int, default=640,
                    help='inference size (pixels)')
parser.add_argument('--augment', action='store_true',
                    help='augmented inference')
parser.add_argument('--update', action='store_true',
                    help='update all models')
parser.add_argument('--task', type = str, default = '',
                    help='update all models')
parser.add_argument('--project', default='runs/detect',
                    help='save results to project/name')
parser.add_argument('--name', default='exp',
                    help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true',
                    help='existing project/name ok, do not increment')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [Id for Id in range(torch.cuda.device_count())]

DMSHN = DMSHN().to(device)
DMSHN= torch.nn.DataParallel(DMSHN, device_ids=device_ids)

save_dir = Path(increment_path(Path(opt.project) / opt.name,
                               exist_ok=opt.exist_ok))  # increment run
def detect(opt , task):
    source, imgsz = opt.source, opt.img_size
    if task == 'derain':
        DMSHN.load_state_dict(torch.load('DMSHN_derain.pth'))
    if task == 'denosiy':
        DMSHN.load_state_dict(torch.load('DMSHN_denoisy.pth'))
    if task == 'dehaze':
        DMSHN.load_state_dict(torch.load('DMSHN_dehaze.pth'))
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    with torch.no_grad():
        DMSHN.eval()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).float()
            img = img.to(device)
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            clean, _ = DMSHN(img)
            clean = clean.cpu().numpy()
            clean = clean.squeeze(0).transpose(1, 2, 0)

            clean = clean * 255
            clean = clean[:, :, ::-1]
            cv2.imwrite(str(save_dir) + '.jpg', clean)

if __name__ == '__main__':
    # check_requirements(exclude=('pycocotools', 'thop'))
    detect(opt, opt.task)
