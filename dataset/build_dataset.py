import shutil
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset
from util.utils import *
from torchvision.datasets import ImageFolder

def DMD(args):
    train_TF = get_transform('train')
    test_TF = get_transform('test')

    divide_subject(test_subject = args.test_subject)
    train_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/train', transform = train_TF)
    test_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/test', transform = test_TF)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_dataloader, test_dataloader

def divide_subject(test_subject = 1):
    '''
    divide subject into train/test (4:1).
    all body images are saved in 

    file_name = [path,file_name]
    file_info = ['body',{time},{subject},{interval count},{frame count}]
    '''
    path = '/data/DMD-Driver-Monitoring-Dataset/'
    train_imgs_list = []
    test_imgs_list = []

    # Read .txt file
    file = open('/data/DMD-Driver-Monitoring-Dataset/body_imgs_list.txt','r')
    body_string = file.read()
    file.close()

    # Extracts the file which include 'driver actions' in file name
    body_file_list = body_string.split('\n')
    for path_file_name in body_file_list:
        if 'driver_actions' in path_file_name:
            file_name = path_file_name.split('/')[-1]
            file_info = file_name.split('_')
            if int(file_info[2])==test_subject:
                test_imgs_list.append(path_file_name)
            else:
                train_imgs_list.append(path_file_name)
    
    if os.path.isfile(path+'check_subject.txt'):
        check_subject = open(path+'check_subject.txt','r')
        subject_num = check_subject.read()
        check_subject.close()
        # if already existing test subject's index == newly introduced subject's index, do nothing
        # if not, clear tree
        if int(subject_num) != test_subject:
            if os.path.exists(path+'train/'):
                shutil.rmtree(path+'train/')
            if os.path.exists(path+'test/'):
                shutil.rmtree(path+'test/')
            check_subject = open(path+'check_subject.txt','w')
            check_subject.write(str(test_subject))
            check_subject.close()
        else:
            return 0
    else:
        check_subject = open(path+'check_subject.txt','w')
        check_subject.write(str(test_subject))
        check_subject.close()

    # copy imgs into the dirs
    build_ImageFolder_shape(train_imgs_list, path, division = 'train')
    build_ImageFolder_shape(test_imgs_list, path, division = 'test')

def build_ImageFolder_shape(imgs_list, path, division = 'train'):
    for file in imgs_list:
        file_label = file.split('/')[-2]

    # merge labels
        if file_label in ['phonecall_left','phonecall_right']:
            file_label = 'phonecall'
        if file_label in ['texting_left','texting_right']:
            file_label = 'texting'
        if file_label in ['unclassified']:
            continue
        if not os.path.exists(path+'/'+division+'/'+file_label):
            os.makedirs(path+'/'+division+'/'+file_label)
        shutil.copy(file,path+'/'+division+'/'+file_label)