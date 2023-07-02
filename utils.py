import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
import torchvision


def parse_args():
    desc = 'Pytorch Implementation of \'Testing code for Blind Image Inpainting via Omni-dimensional Gated Attention and Wavelet Queries\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path_test', type=str, default='./datasets/')
    parser.add_argument('--task_name', type=str, default='inpaint', choices=['inpaint'])
    parser.add_argument('--dataset_name', type=str, default='places')
    parser.add_argument('--model_file', type=str, default='./checkpoints/places.pth', help='path of pre-trained model file')
    parser.add_argument('--num_iter', type=int, default=700000, help='iterations of training')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')    
    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path_test = args.data_path_test
        self.task_name = args.task_name
        self.dataset_name = args.dataset_name
        self.num_iter = args.num_iter
        self.workers = args.workers
        self.model_file = args.model_file


def init_args(args):
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)


def pad_image_needed(img, size):
    width, height = T._get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img


class TestDataset(Dataset):
    def __init__(self, data_path, data_path_test, task_name, data_type, length=None):
        super().__init__()
        self.task_name, self.data_type= task_name, data_type

        self.corrupted_images_test = sorted(glob.glob('{}/input/*'.format(data_path_test)))
        self.target_images_test = sorted(glob.glob('{}/target/*'.format(data_path_test)))

        self.num = len(self.corrupted_images_test)
        self.num_test = len(self.corrupted_images_test)
        self.sample_num = length if data_type == 'train' else self.num_test

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        corrupted_image_name = os.path.basename(self.corrupted_images_test[idx % self.num])
        target_image_name = os.path.basename(self.target_images_test[idx % self.num])

        if self.task_name=='inpaint':
            corrupted = T.to_tensor((Image.open(self.corrupted_images_test[idx % self.num])).resize((256,256)))
            target = T.to_tensor((Image.open(self.target_images_test[idx % self.num])).resize((256,256)))
        else: 
            corrupted = T.to_tensor((Image.open(self.corrupted_images_test[idx % self.num])).resize((256,256)))
            target = T.to_tensor((Image.open(self.target_images_test[idx % self.num])).resize((256,256)))        

        
        h, w = corrupted.shape[1:]
        return corrupted, target, corrupted_image_name, h, w
