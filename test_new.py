import argparse
import torch
from torchvision import transforms

import opt
from places2_new import Places2_new
from evaluation_new import evaluate_new
from net import PConvUNet
from util.io import load_ckpt







parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./ws/image/*.jpg')
#parser.add_argument('--root', type=str, default='./srv/datasets/Places2/val_ijk/**/*.jpg')
parser.add_argument('--snapshot', type=str, default='./snapshots')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--mask_root', type=str, default='./ws/mask_ws')
args = parser.parse_args()

device = torch.device('cuda')
size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


print("mask_transform",mask_transform)
#dataset_val = Places2(args.root, img_transform, mask_transform, 'val')
dataset_val = Places2_new(args.root, args.mask_root, img_transform, mask_transform, 'ws')
print("ws ",dataset_val)

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
print("dataset_val type",type(dataset_val))
print("dataset_val",dataset_val)
evaluate_new(model, dataset_val, device, './ws/inpaint/result.jpg')
