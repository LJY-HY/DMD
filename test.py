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


    _, test_dataloader = globals()[args.test_dataset](args)

    # Get architecture
    net = get_architecture(args)
    net = net.to(args.device)

    CE_loss = torch.nn.CrossEntropyLoss()
    # path = './checkpoint/'+args.arch+'_'+args.train_dataset+'_'+args.freeze+'_'+args.optimizer+'.pth'
    path = './checkpoint/'+args.arch+'_'+args.train_dataset+'_freeze_'+args.optimizer+'_'+str(args.freeze)+'.pth'
    result = './checkpoint/'+args.arch+'_'+args.train_dataset+'.txt'
    
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
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
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
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    print('Accuracy :'+ '%0.4f'%acc )
    return acc

if __name__ == '__main__':
    main()