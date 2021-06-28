import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import argparse

from util.arguments import get_arguments
from util.utils import *


def DMD(args):
    TF = transforms.Compose([
        transforms.ToTensor()
        ])
        
    train_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/train', transform = TF)
    test_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/test', transform = TF)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    return train_dataloader, test_dataloader

def main():
    # argument parsing
    # --gpu_id 0
    # --dataset DMD
    # --arch MobileNet
    # --batch_size 128
    # --trial 0

    path_DMD = '/data/DMD-Driver-Monitoring-Dataset/'

    args = argparse.ArgumentParser()
    args = get_arguments()
    args.num_classes = 11

    # Get Dataset
    train_dataloader, _ = globals()[args.dataset](args)
    mean, std = get_DMD_info(train_dataloader, force=True)
    print(f'Get mean and std on {args.dataset} dataset')

    print(f'{mean},{std}')


if __name__ == '__main__':
    main()