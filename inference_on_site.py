import cv2
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
from PIL import Image
from util.utils import *
from dataset.build_DMD_deployment import DMD_deployment
from typing import Any

gt_number = [43,351,386,1822,175,95,1084,2476,0,87,741]          # Ground truth of subject #1
gt =[]
for idx,number in enumerate(gt_number):
    temp = [idx for _ in range(number)]
    gt = gt+temp

def main():
    path_DMD = '/data/DMD-Driver-Monitoring-Dataset/'

    args = argparse.ArgumentParser()
    args.gpu_id = 0
    args.dataset = 'DMD_deployment'
    args.test_subject=1
    args.arch = 'ResNet50'
    args.optimizer= 'SGD'
    args.scheduler = 'MultiStepLR'
    args.batch_size=1
    args.freeze=0.75

    args.device = torch.device('cuda',args.gpu_id)

    args.num_classes = 11

    # Get architecture
    args.gpu_id=0
    args.device = torch.device('cuda',args.gpu_id)
    net_CLC = get_architecture(args)
    net_CLC = net_CLC.to(args.device)

    args.device = torch.device('cuda',args.gpu_id)
    net_base = get_architecture(args)
    net_base = net_base.to(args.device)

    CE_loss = torch.nn.CrossEntropyLoss()
    # name ='./checkpoint/'+args.arch+'_'+args.train_dataset+'_freeze_'+str(args.freeze)
    path_CLC = '/home/esoc/LeeJaeyoon/DMD/checkpoint/deployment/ResNet50_deployment_on_DMD_threshold_0.05_often_10_im.pth'
    path_base = '/home/esoc/LeeJaeyoon/DMD/checkpoint/ResNet50_DMD_freeze_0.75.pth'
  
    # Load checkpoint
    state_dict_CLC = torch.load(path_CLC)
    state_dict_base= torch.load(path_base)
    net_CLC.load_state_dict(state_dict_CLC)
    net_base.load_state_dict(state_dict_base)

    acc=0
    best_train=0
    train_best = 0
    output_ = test(args, net_CLC, net_base)
    
def eval_compare(args,frame,frame_count,net_CLC,net_base):
    # transform
    mean_temp = (0.5,0.5,0.5)
    std_temp = (0.25,0.25,0.25)
    normalize = transforms.Normalize(mean = mean_temp, std = std_temp)
    TF = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
        ])
        
    with torch.no_grad():
            new_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame_to_Image = Image.fromarray(new_frame)
            frame_to_Image = frame_to_Image.convert('RGB')
            input_frame = TF(frame_to_Image)
            input_frame = input_frame.to(args.device)
            input_frame = input_frame.unsqueeze(0)
           
            outputs_clc = net_CLC(input_frame)
            outputs_base = net_base(input_frame)

            pred_clc = outputs_clc.argmax(dim=1)
            pred_base = outputs_base.argmax(dim=1)
    return pred_base.item(), pred_clc.item(), gt[frame_count]

def test(args, net_CLC, net_base):
    net_CLC.eval()
    net_base.eval()

    # Get Video
    video_path = '/data/DMD-Driver-Monitoring-Dataset/dmd_test.mp4'
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        print('No Video')
        exit()
    
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # 1280
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 720
    frame_size = (frameWidth,frameHeight)
    frameRate = 25
    frame_count=0
    while True:
        retval,frame = cap.read()
        if not(retval):
            break
        pred_base,pred_clc,gt = eval_compare(args,frame,frame_count,net_CLC,net_base)
        # print(pred_base,pred_clc,gt)
        frame_count+=1

if __name__ == '__main__':
    main()