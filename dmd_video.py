import cv2
import numpy as np

from util.arguments import get_arguments_test
from util.utils import *
from PIL import Image
import torch
import torchvision.transforms as transforms

import argparse
import time


classes = ['change_gear','drinking','hair_and_makeup',
           'phonecall','radio','reach_backseat',
           'reach_side','safe_drive','standstill_or_waiting',
           'talking_to_passenger','texting']

gt_number= [43,351,386,1822,175,95,1084,2476,0,87,741]
gt_list =[-1]
for idx,number in enumerate(gt_number):
    temp = [idx for _ in range(number)]
    gt_list = gt_list+temp

# text font setting
font                   = cv2.FONT_HERSHEY_DUPLEX 
bottomLeftCornerOfText = (10,500)
fontScale              = 0.7
fontBlack              = (0,0,0)
fontRed                = (0,0,255)
fontBlue               = (255,0,0)
fontGreen              = (0,255,0)
lineType               = 0


def main():

    args = argparse.ArgumentParser()
    args = get_arguments_test()
    args.num_classes = 11
    args.batch_size=1
    args.arch='ResNet50'
    args.device_base = torch.device('cuda',0)
    args.device_clc = torch.device('cuda',1)

    # Read video
    cap = cv2.VideoCapture('./dmd_test.mp4')
    text_area = np.ones((450,500,3),dtype=np.uint8)*255

    cv2.putText(text_area,'Ground Truth', (170,50), font, fontScale, fontBlack, lineType)
    cv2.putText(text_area,'Baseline', (20,150),font, fontScale, fontBlack, lineType)
    cv2.putText(text_area,'Prediction:',(20, 180),font,fontScale,fontBlack, lineType)
    cv2.putText(text_area,'ACC:',(20, 200),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,'Inference Time:',(20, 220),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,'ms',(400, 220),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,"Avg Inference Time:",(20,240),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,'ms',(400, 240),font,fontScale,fontBlack,lineType)

    cv2.putText(text_area,'Ours', (20, 260),font, fontScale, fontBlack, lineType)
    cv2.putText(text_area,'Prediction:',(20, 290),font,fontScale,fontBlack, lineType)
    cv2.putText(text_area,'ACC:',(20, 310),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,'Inference Time:',(20, 330),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,'ms',(400, 330),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,"Avg Inference Time:",(20,350),font,fontScale,fontBlack,lineType)
    cv2.putText(text_area,'ms',(400, 350),font,fontScale,fontBlack,lineType)

    cv2.putText(text_area,'Frames:',(20, 370),font,fontScale,fontBlack,lineType)

    # Get architecture
    # base model
    path = 'checkpoint/ResNet50_DMD_freeze_0.75.pth'
    args.device = args.device_base
    net_base = get_architecture(args)
    state_dict = torch.load(path)
    net_base.load_state_dict(state_dict)
    net_base = net_base.to(args.device_base)
    net_base.eval()

    # modified model(ours)
    path = 'checkpoint/deployment/ResNet50_deployment_on_DMD_threshold_0.05_often_10_im.pth'
    args.device = args.device_clc
    net_clc = get_architecture(args)
    state_dict = torch.load(path)
    net_clc.load_state_dict(state_dict)
    net_clc = net_clc.to(args.device_clc)
    net_clc.eval()

    # Count
    frame_count = 0
    base_acc = 0
    clc_acc = 0
    base_avg_time = 0
    clc_avg_time = 0

    # For video save
    #video_path = './demo.mp4'
    #fps = 25
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter(video_path, fourcc, fps, (1300,450))

    # Show video
    while (cap.isOpened()) :
        frame_count += 1
        if frame_count > 7260 : break
        if gt_list[frame_count] == -1 : continue
        #print('proceeding...',frame_count)

        # Video stream
        ret, frame = cap.read()
        if not(ret): break

        # Inference
        res_base, res_clc, label,clc_time, base_time = eval_compare(args,frame,frame_count,net_clc,net_base)
        clc_time = round(clc_time*1000,3)
        base_time = round(base_time*1000,3)
        
        frame = cv2.resize(frame,dsize=(800,450))
    
        cv2.namedWindow('results')     # have to ignore when video saving
        frame = cv2.hconcat([frame,text_area])

        # Text Results
        # correct = Blue, incorrect = Red
        # Ground Truth
        cv2.putText(frame,classes[label], (1000,80),font,fontScale, fontBlue, lineType)

        # Base Model
        # Prediction
        cv2.putText(frame,classes[res_base],(1030,180),font,fontScale, fontBlue if label == res_base else fontRed,lineType)
        # Acc
        if res_base == label: base_acc += 1
        cv2.putText(frame,str(round(base_acc/frame_count*100,2))+'%', (1050,200), font, fontScale, fontBlack, lineType)
        cv2.putText(frame,str(base_time),(1100, 220),font,fontScale,fontBlack,lineType)
        base_avg_time += base_time
        cv2.putText(frame,str(round(base_avg_time/frame_count,3)),(1100, 240),font,fontScale,fontBlack,lineType)
        
        # Modified Model
        # Prediction
        cv2.putText(frame,classes[res_clc],(1030,290),font,fontScale, fontBlue if label == res_clc else fontRed, lineType)
        # Acc
        if res_clc == label : clc_acc += 1
        cv2.putText(frame,str(round(clc_acc/frame_count*100,2))+'%',(1050,310),font,fontScale, fontBlack,lineType)
        cv2.putText(frame,str(clc_time),(1100, 330),font,fontScale,fontBlack,lineType)
        clc_avg_time += clc_time
        cv2.putText(frame,str(round(clc_avg_time/frame_count,3)),(1100, 350),font,fontScale,fontBlack,lineType)

        # Frame
        cv2.putText(frame,str(frame_count),(1100, 370),font,fontScale,fontBlack,lineType)

        # Frame show
        cv2.imshow('results',frame)

        # Frame save
        #out.write(frame)

        if cv2. waitKey(10) == 27 : break
    #out.release()
    cap.release()
    cv2.destroyAllWindows()

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
            #input_frame = input_frame.to(args.device)
            input_frame = input_frame.unsqueeze(0)

            start_time = time.time()
            outputs_clc = net_CLC(input_frame.to(args.device_clc))
            clc_time = time.time()
            outputs_base = net_base(input_frame.to(args.device_base))
            base_time = time.time()

            base_time -= clc_time
            clc_time -= start_time

            pred_clc = outputs_clc.argmax(dim=1)
            pred_base = outputs_base.argmax(dim=1)
    return pred_base.cpu().item(), pred_clc.cpu().item(), gt_list[frame_count], clc_time, base_time

if __name__ == '__main__':
    main()