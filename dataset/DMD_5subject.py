import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset
from utils.utils import *

def DMD(args):
    train_TF = get_transform('train')
    test_TF = get_transform('test')

    train_dataset = 
    
    return train_dataloader, test_dataloader