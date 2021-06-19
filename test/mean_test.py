import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse

from util.arguments import get_arguments
from util.utils import *
from dataset.build_DMD import DMD


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
    mean, std = get_DMD_info(path_DMD, train_dataloader)
    print(f'Get mean and std on {args.dataset} dataset')

    print(f'{mean},{std}')


if __name__ == '__main__':
    main()