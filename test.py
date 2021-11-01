from dataset.build_StateFarm import StateFarm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse

from util.arguments import get_arguments_test
from util.utils import *
from dataset.build_DMD import DMD
from dataset.build_StateFarm import StateFarm
from dataset.build_DMD_deployment import DMD_deployment

def main():
    # argument parsing
    # --gpu_id 0
    # --dataset DMD
    # --arch MobileNet
    # --batch_size 128
    # --trial 0

    path_DMD = '/data/DMD-Driver-Monitoring-Dataset/'
    path_StateFarm = '/data/driver_detection/'

    args = argparse.ArgumentParser()
    args = get_arguments_test()
    args.device = torch.device('cuda',args.gpu_id)

    args.num_classes = 11

    # Get Dataset
    if args.test_dataset == 'DMD':
        if os.path.isfile(path_DMD+'check_subject.txt'):
            check_subject = open(path_DMD+'check_subject.txt','r')
            subject_num = check_subject.read()
            check_subject.close()
        args.test_subject = int(subject_num)

    if args.test_dataset=='DMD_deployment':
        args.dataset = args.test_dataset
    _, _, test_dataloader = globals()[args.test_dataset](args)

    # Get architecture
    net = get_architecture(args)
    net = net.to(args.device)

    CE_loss = torch.nn.CrossEntropyLoss()
    name ='./checkpoint/'+args.arch+'_'+args.train_dataset+'_freeze_'+str(args.freeze)
    if args.option is not None:
        path = name+'_'+args.option+'.pth'
        result = name+'_'+args.option+'.txt'
    else:
        path = name+'.pth'
        result = name+'.txt'
    path = 'checkpoint/deployment/ResNet50_deployment_on_DMD_threshold_0.05_often_10_im.pth'
    print(path)
    print(result)
    
    # Load checkpoint
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)

    acc=0
    best_train=0
    train_best = 0
    acc = test(args, net, test_dataloader)
    
    import sys
    sys.stdout = open(result,'a')
    print('Test on {} dataset'.format(args.test_dataset))
    print('Train Acc at best acc:', best_train)
    print('Best Train Acc:', train_best)
    print('Last Acc:', acc)

def test(args, net, test_dataloader):
    net.eval()
    output_labels = []
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, items in enumerate(test_dataloader):
            if len(items)==3:
                inputs, targets, index = items
            else:
                inputs, targets = items
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
            output_labels+=outputs.argmax(dim=1).tolist()
    p_bar.close()
    f = open('/home/esoc/LeeJaeyoon/DMD/ImgsName_TrueLabel_OutputLabel','w')
    for i in range(len(test_dataloader.dataset)):
        ImgsName,TrueLabel = test_dataloader.dataset.samples[i]
        Line = str(ImgsName)+' '+str(TrueLabel)+' '+str(output_labels[i])+'\n'
        f.write(Line)
    f.close()
    acc = acc/test_dataloader.dataset.__len__()
    print('Accuracy :'+ '%0.4f'%acc )
    return acc

if __name__ == '__main__':
    main()