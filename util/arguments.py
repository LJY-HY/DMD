import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='DMD', type=str, choices=['cifar10','DMD','StateFarm'])
    parser.add_argument('--arch', default = 'MobileNet', type=str, choices = ['Inception','MobileNetv2','MobileNetv3','ShuffleNet','ResNet34','ResNet50'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr', default = 0.001, type=float, choices = [1.0,0.1,0.01,0.001,0.0005,0.0002,0.0001])
    parser.add_argument('--epoch', default=40, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=64, type=int, choices=[32,64,128])
    parser.add_argument('--test_subject', default=1, type = int, choices=[0,1,2,3,4])
    parser.add_argument('--dropout_rate', default=0.5, type=float, choices=[0,0.3,0.5,0.7])
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-6])
    parser.add_argument('--warmup_duration', default = 10, help = 'duration of warming up')
    parser.add_argument('--trial', default = '0', type=str)
    parser.add_argument('--freeze', default=0.75, type=float, help = 'freeze rate of pretrained network')
    parser.add_argument('--option', default='', type=str, help='path naming option')
    args = parser.parse_args()
    return args

def get_arguments_test():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--train_dataset', default='DMD', type=str, choices=['DMD','StateFarm'])
    parser.add_argument('--test_dataset',default = 'DMD', type = str, choices = ['DMD','StateFarm'])
    parser.add_argument('--arch', default = 'MobileNetv2', type=str, choices = ['Inception','MobileNetv2','MobileNetv3','ShuffleNet','ResNet34','ResNet50'])
    parser.add_argument('--batch_size', default=128, type=int, choices=[32,64,128])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--trial', default = '0', type=str)
    parser.add_argument('--freeze', default = 0.75, type=float, help = 'freeze rate of pretrained network')
    parser.add_argument('--option', default=None, type=str, help='you can give tilt')
    args = parser.parse_args()
    return args

# 이거 쓰이나?
# def get_arguments_target():
#     parser = argparse.ArgumentParser(description = 'Training Arguments')
#     parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
#     parser.add_argument('--pretrain_dataset', default='DMD', type=str, choices=['DMD','StateFarm'])
#     parser.add_argument('--dataset',default = 'StateFarm', type = str, choices = ['DMD','StateFarm'])
#     parser.add_argument('--arch', default = 'MobileNet', type=str, choices = ['Inception','MobileNetv2','MobileNetv3','ShuffleNet','ResNet34','ResNet50'])
#     parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
#     parser.add_argument('--lr', default = 0.001, type=float, choices = [1.0,0.1,0.01,0.001,0.0005,0.0002,0.0001])
#     parser.add_argument('--epoch', default=40, type=int, help='number of total epochs')
#     parser.add_argument('--batch_size', default=64, type=int, choices=[32,64,128])
#     parser.add_argument('--test_subject', default=1, type = int, choices=[0,1,2,3,4])
#     parser.add_argument('--dropout_rate', default=0.5, type=float, choices=[0,0.3,0.5,0.7])
#     parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
#     parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-6])
#     parser.add_argument('--warmup_duration', default = 10, help = 'duration of warming up')
#     parser.add_argument('--trial', default = '0', type=str)
#     parser.add_argument('--freeze', default=0.75, type=float, help = 'freeze rate of pretrained network')
#     args = parser.parse_args()
#     return args

def get_arguments_deploy():
    parser = argparse.ArgumentParser(description = 'Deployment Scenario')
    parser.add_argument('--deployment_subject',default='p002', type = str, help='[2,12,14,15,16,21,22,24,26,35,39,41,42,45,47,49,50,51,52,56,61,64,66,72,75,81]')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='StateFarm', type=str)
    parser.add_argument('--arch', default = 'ResNet50', type=str, choices = ['ResNet50'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr', default = 0.001, type=float, choices = [1.0,0.1,0.01,0.001,0.0005,0.0002,0.0001])
    parser.add_argument('--epoch', default=40, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=64, type=int, choices=[32,64,128])
    parser.add_argument('--dropout_rate', default=0.5, type=float, choices=[0,0.3,0.5,0.7])
    parser.add_argument('--finetuning', action = 'store_true')
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-6])
    parser.add_argument('--warmup_duration', default = 10, help = 'duration of warming up')
    parser.add_argument('--correction',action = 'store_true')
    parser.add_argument('--freeze', default=0.75, type=float, help = 'freeze rate of pretrained network')
    parser.add_argument('--option', default='', type=str, help='path naming option')
    parser.add_argument('--correction_th',type = float,default = 0.05,choices = [0.05,0.03,0.01,0.00])
    parser.add_argument('--im',action = 'store_true')
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args

def get_t_SNE_arguments():
    parser = argparse.ArgumentParser(description = 'Drawing t-SNE')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='StateFarm', type=str)
    parser.add_argument('--arch', default = 'ResNet50', type=str)
    parser.add_argument('--batch_size', default=64, type=int, choices=[64,128,256,512])
    parser.add_argument('--deployment_subject',default='p002', type = str, help='[2,12,14,15,16,21,22,24,26,35,39,41,42,45,47,49,50,51,52,56,61,64,66,72,75,81,all]')
    parser.add_argument('--test_subject', default=1, type = int, choices=[0,1,2,3,4], help = 'used when drawing DMD')
    parser.add_argument('--perplexity', default = 10, type = int)
    parser.add_argument('--freeze', default=0.75, type=float, help = 'freeze rate of pretrained network')
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args