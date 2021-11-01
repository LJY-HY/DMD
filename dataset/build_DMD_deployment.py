import shutil
import os
from torch.utils.data import DataLoader
from util.utils import *
from dataset.ImageFolder import ImageFolder
from PIL import Image

'''
Num of Class : 11
classes =  ['change_gear',
            'drinking',
            'hair_and_makeup',
            'phonecall_left',       ---> 'phonecall'
            'phonecall_right',      ---> 'phonecall'
            'radio',
            'reach_backseat',
            'reach_side',
            'safe_drive',
            'standstill_or_waiting',
            'talking_to_passenger',
            'texting_left',         ---> 'texting'
            'texting_right',        ---> 'texting'
            'unclassified'          ---> deprecated
            ]

1) 'phonecall_left' and 'phonecall_right' are combined as phonecall
2) 'texting_left' and 'texting_right' are combined as texting
3) 'unclassified' is not considered as a single class
'''

def DMD_deployment(args):
    '''
    Similar to DMD.
    However, this dataset is for deployment only with DMD dataset.
    Therefore, subject #1 used for test in 'DMD' is further divided into train/test dataset for deployment.
    Other subjects are not accessible while deployment.    
    '''
    train_TF = get_transform('train')
    test_TF = get_transform('test')
    '''
    data shape : [bsz,depth,height,width]
    '''
    divide_subject_by_ratio(args,test_subject = args.test_subject)
    train_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/train_DMD_deployment', transform = train_TF)
    val_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/test_DMD_deployment', transform = test_TF)
    test_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/test_DMD_deployment', transform = test_TF)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 16)
    return train_dataloader, val_dataloader, test_dataloader


def divide_subject_by_ratio(args,test_subject = 1):
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
    if args.dataset!='DMD_deployment':
        if os.path.isfile(path+'check_subject.txt'):
            check_subject = open(path+'check_subject.txt','r')
            subject_num = check_subject.read()
            check_subject.close()
            if int(subject_num) != test_subject:
                if os.path.exists(path+'train_DMD_deployment/'):
                    shutil.rmtree(path+'train_DMD_deployment/')
                if os.path.exists(path+'test_DMD_deployment/'):
                    shutil.rmtree(path+'test_DMD_deployment/')
                check_subject = open(path+'check_subject.txt','w')
                check_subject.write(str(test_subject))
                check_subject.close()
            else:
                return 0
        else:
            check_subject = open(path+'check_subject.txt','w')
            check_subject.write(str(test_subject))
            check_subject.close()
   
    imgs_list_dict = {}
    for file in test_imgs_list:
        file_label = file.split('/')[-2]
        # merge labels
        if file_label in ['phonecall_left','phonecall_right']:
            file_label = 'phonecall'
        if file_label in ['texting_left','texting_right']:
            file_label = 'texting'
        if file_label in ['unclassified']:
            continue
        if file_label in imgs_list_dict:
            imgs_list_dict[file_label].append(file)
        else:
            imgs_list_dict[file_label] = [file]
    
    for label in imgs_list_dict:
        train_set = imgs_list_dict[label][:int(0.8*len(imgs_list_dict[label]))]
        test_set = imgs_list_dict[label][int(0.8*len(imgs_list_dict[label])):]
        if not os.path.exists(path+'/train_DMD_deployment/'+label):
            os.makedirs(path+'/train_DMD_deployment/'+label)
        for train_data in train_set:
            shutil.copy(train_data,path+'/train_DMD_deployment/'+label)
        if not os.path.exists(path+'/test_DMD_deployment/'+label):
            os.makedirs(path+'/test_DMD_deployment/'+label)
        for test_data in test_set:
            shutil.copy(test_data,path+'/test_DMD_deployment/'+label)
    if not os.path.exists(path+'/train_DMD_deployment/standstill_or_waiting'):
            os.makedirs(path+'/train_DMD_deployment/standstill_or_waiting')
    if not os.path.exists(path+'/test_DMD_deployment/standstill_or_waiting'):
            os.makedirs(path+'/test_DMD_deployment/standstill_or_waiting')