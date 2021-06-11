import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

import os
from tqdm import tqdm
import argparse

from util.arguments import get_arguments
from util.utils import *
from dataset.build_DMD import DMD

def get_cifar10(args):
    mean = (0.5071, 0.4865, 0.4408)
    std = (0.2673, 0.2564, 0.2762)
    path = '~/datasets/cifar10'

    transform_train = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            #transforms.Resize((300,300)),       #Inception
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
            ])

    transform_test = transforms.Compose([
        #transforms.Resize((300,300)),       #Inception
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        ])

    train_loader = DataLoader(
            datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train),
            batch_size=256, shuffle=False
    )
    test_loader = DataLoader(
            datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test),
            batch_size=500, shuffle=False
    )
    return train_loader, test_loader
    

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()
    args.device = torch.device('cuda',args.gpu_id)

    args.num_classes= 13 if args.dataset=='DMD' else 10

    # Get Dataset
    train_dataloader, test_dataloader = DMD(args) if args.dataset=='DMD' else get_cifar10(args)

    # Get architecture
    net = get_architecture(args)
    net = net.to(args.device)

    # Get optimizer, scheduler
    optimizer, scheduler = get_optim_scheduler(args,net)
       
    CE_loss = torch.nn.CrossEntropyLoss()
    training = ''
    path = './checkpoint/'+args.arch+'_'+args.dataset+'.pth'
    result = './checkpoint/'+args.arch+'_'+args.dataset+'.txt'
    
    best_acc=0
    acc=0
    best_train=0
    train_best = 0
    for epoch in range(args.epoch):
        train_acc,_ = train(args, net, train_dataloader, optimizer, scheduler, CE_loss, epoch)
        print('train_acc:',train_acc)
        acc = test(args, net, test_dataloader, optimizer, scheduler, CE_loss, epoch)
        scheduler.step()

        if train_acc > train_best:
            train_best = train_acc

        if best_acc<acc:
            best_acc = acc
            best_train = train_acc
            torch.save(net.state_dict(), path)
    
    import sys
    sys.stdout = open()
    print('Best Acc:', best_acc)
    print('Train Acc at best acc:', best_train)
    print('Best Train Acc:', train_best)
    print('Last Acc:', acc)

def train(args, net, train_dataloader, optimizer, scheduler, CE_loss, epoch):
    net.train()
    train_loss = 0
    acc = 0
    p_bar = tqdm(range(train_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)     

        if args.arch == 'Inception':
            outputs,_ = net(inputs)
        else :
            outputs = net(inputs)

        loss = CE_loss(outputs,targets)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        acc += sum(outputs.argmax(dim=1)==targets)
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_dataloader.__len__(),
                    lr=scheduler.optimizer.param_groups[0]['lr'],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    
    return acc/train_dataloader.dataset.__len__(), train_loss/train_dataloader.__len__()        # average train_loss

def test(args, net, test_dataloader, optimizer, scheduler, CE_loss, epoch):
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
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    lr=scheduler.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    print('Accuracy :'+ '%0.4f'%acc )
    return acc


if __name__ == '__main__':
    main()

# TODO : combine model saving/loading method