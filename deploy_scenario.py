import sys
import torch
import torch.nn.functional as F

from tqdm import tqdm
import argparse

from util.arguments import get_arguments_deploy
from util.utils import *
from dataset.build_StateFarm import StateFarm
from dataset.build_DMD_deployment import DMD_deployment

def main():
    original_stdout = sys.stdout
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments_deploy()
    args.device = torch.device('cuda',args.gpu_id)
    torch.cuda.set_device(args.device)
    args.num_classes = 11
    
    # Get Dataset
    train_dataloader, val_dataloader, test_dataloader = globals()[args.dataset](args)

    # Get architecture
    net = get_architecture(args)
    
    net = net.to(args.device)
    
    name ='./checkpoint/'+args.arch+'_DMD_freeze_'+str(args.freeze)+'.pth'
    state_dict = torch.load(name)
    net.load_state_dict(state_dict)
    net.to(args.device)

    # Get optimizer, scheduler
    optimizer, scheduler = get_optim_scheduler(args,net)
       
    CE_loss = torch.nn.CrossEntropyLoss()

    dir_path = './checkpoint/deployment'
    if args.correction:
        dir_path = dir_path + '_CLC'
    elif args.finetuning:
        dir_path = dir_path + '_finetuning'

    path = dir_path+'/'+args.arch+'_deployment_on_threshold'+str(args.correction_th)+'_'+args.deployment_subject+'.pth'
    result = dir_path+'/'+args.arch+'_deployment_on_threshold'+str(args.correction_th)+'_'+args.deployment_subject+'.txt'
    if args.dataset == 'DMD_deployment':
        name = dir_path+'/'+args.arch+'_deployment_on_DMD_threshold_'+str(args.correction_th)+'_often_'+str(args.often)
        if args.im:
            name = name+'_im'
        name = name + str(args.trial)
        result = result + str(args.trial)
        path = name +'.pth'
        result = name + '.txt'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    best_train_acc = 0

    best_val_acc=0
    test_acc_at_best_val_acc = 0
    train_acc_at_best_val_acc=0

    best_test_acc = 0

    # Check the initial accuracy on DMD trained model
    for epoch in range(1):
        pre_acc = test(args, net, test_dataloader, scheduler,'Test')
        with open(result,'a') as f:
            sys.stdout = f
            print('Test Accuracy with no other adaptation') 
            print('Test Accuracy before any action: {:.2f}%\n'.format(100*pre_acc.item()))
            sys.stdout = original_stdout

    if not args.finetuning:
        # Labeling StateFarm dataset through DMD trained model
        for epoch in range(1):
            with torch.no_grad():
                for batch_idx, (inputs, targets, index) in enumerate(train_dataloader):
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                    outputs = net(inputs)
                    corrected_labels = outputs.argmax(dim=1)
                    for num, idxs in enumerate(index):
                        train_dataloader.dataset.samples[idxs] = list(train_dataloader.dataset.samples[idxs])
                        train_dataloader.dataset.samples[idxs][1] = corrected_labels[num].item()
                        train_dataloader.dataset.samples[idxs] = tuple(train_dataloader.dataset.samples[idxs])
        print('Labeling Done!!')
    # Training
    for epoch in range(args.epoch):
        train_acc,_ = train(args, net, train_dataloader, optimizer, scheduler, CE_loss, epoch)
        print('train_acc:',train_acc)
        val_acc = test(args, net, val_dataloader, scheduler,'Validation')
        test_acc = test(args, net, test_dataloader, scheduler,'Test')
        scheduler.step()
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if best_val_acc<val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val_acc = test_acc
            train_acc_at_best_val_acc = train_acc
            torch.save(net.state_dict(), path)

    sys.stdout = open(result,'a')
    print('Best Train Acc: {:.2f}%'.format(100*best_train_acc.item()))
    print('Best Test  Acc: {:.2f}%'.format(100*best_test_acc.item()))
    print('Best Validation Acc: {:.2f}%'.format(100*best_val_acc.item()))
    print('Test  Acc at Best Validation Acc: {:.2f}%'.format(100*test_acc_at_best_val_acc.item()))
    print('Train Acc at Best Validation Acc: {:.2f}%'.format(100*train_acc_at_best_val_acc.item()))
    print('Last Acc: {:.2f}%'.format(100*test_acc.item()))

def train(args, net, train_dataloader, optimizer, scheduler, CE_loss, epoch):
    net.train()
    train_loss = 0
    acc = 0
    p_bar = tqdm(range(train_dataloader.__len__()))
    loss_average = 0
    XentLoss_ = nn.CrossEntropyLoss(reduction='none')

    temp_distribution = [0 for i in range(11)]
    target_distribution = [0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091]                 # uniform
    # target_distribution = [0.0, 0.103, 0.085, 0.207, 0.103, 0.090, 0.0, 0.111, 0.0, 0.095, 0.206]                     # StateFarm
    # target_distribution = [0.0058, 0.0484, 0.0531, 0.2511, 0.0241, 0.0130, 0.1493, 0.3411, 0.0, 0.0120, 0.1021]         # DMD subject 1
    temp_distribution = torch.tensor(temp_distribution).cuda()
    # pseudo_distribution = torch.tensor(pseudo_distribution).cuda()
    target_distribution = torch.tensor(target_distribution).cuda()

    for batch_idx, (inputs, targets, index) in enumerate(train_dataloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)     
        if args.arch == 'Inception':
            outputs,_ = net(inputs)
        else :
            outputs = net(inputs)
        if args.correction:
            if epoch % args.often==0:
                with torch.no_grad():
                    loss_ = XentLoss_(outputs,targets)
                corrected_labels = torch.where(loss_>sorted(loss_)[int(inputs.shape[0]*(1-args.correction_th* (1-epoch/args.epoch)**2)-1)], outputs.argmax(dim=1), targets)
                for num, idxs in enumerate(index):
                        train_dataloader.dataset.samples[idxs] = list(train_dataloader.dataset.samples[idxs])
                        train_dataloader.dataset.samples[idxs][1] = corrected_labels[num].item()
                        train_dataloader.dataset.samples[idxs] = tuple(train_dataloader.dataset.samples[idxs])
            else:
                corrected_labels = targets
        else:
            corrected_labels = targets
        optimizer.zero_grad()
        loss = CE_loss(outputs,corrected_labels)

        if args.im:
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(torch.sum(-softmax_out*torch.log(softmax_out+1e-5),dim=1))
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5) + msoftmax*torch.log(target_distribution+1e-5))
            entropy_loss-=gentropy_loss
            loss+=entropy_loss

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

def test(args, net, dataloader, scheduler, mode):
    net.eval()
    # output_label = []
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(dataloader):
            import pdb;pdb.set_trace()
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("{mode} Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    mode = mode,
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=dataloader.__len__(),
                    lr=scheduler.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
            # output_label+=outputs.argmax(dim=1).tolist()
    p_bar.close()
    acc = acc/dataloader.dataset.__len__()
    print(mode+' Accuracy :'+ '%0.4f'%(100*acc) )
    return acc


if __name__ == '__main__':
    main()

# TODO : combine model saving/loading method