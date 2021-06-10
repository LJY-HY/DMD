import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default = 'Inception', type=str, choices = ['Inception','MobileNet','ShuffleNet'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr', default = 0.001, type=float, choices = [1.0,0.1,0.001,0.0005,0.0002,0.0001])
    parser.add_argument('--epoch', default=40, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=64, type=int, choices=[32,64])
    parser.add_argument('--dropout_rate', default=0.5, type=float, choices=[0,0.3,0.5,0.7])
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-6])
    parser.add_argument('--warmup_duration', default = 10, help = 'duration of warming up')
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args