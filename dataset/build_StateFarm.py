import csv
import os
import shutil
from torch.utils.data import DataLoader
from util.utils import *
from torchvision.datasets import ImageFolder

def StateFarm(args):
    '''
    Num of Class : 10
    classes =  [c0 : safe driving                   ---> safe_drive
                c1 : texting right                  ---> texting*
                c2 : talking on the phone right     ---> phonecall*
                c3 : texting left                   ---> texting*
                c4 : talking on the phone left      ---> phonecall*
                c5 : operating the radio            ---> radio
                c6 : drinking                       ---> drinking
                c7 : reaching behind                ---> reach_backseat
                c8 : hair and makeup                ---> hair_and_makeup
                c9 : talking to passenger           ---> talking_to_passenger
                ]
    
    1) c1 and c3 will be combined as texting
    2) c2 and c4 witt be combined as talking on the phone
    '''
    
    # set transform
    train_TF = get_transform('train')
    test_TF = get_transform('test')

    train_subject_list, test_subject_list = divide_subject()

    make_ImageFolder(train_subject_list, division = 'train')
    make_ImageFolder(test_subject_list, division = 'test')

    train_dataset = ImageFolder(root = '/data/driver_detection/train', transform = train_TF)
    test_dataset = ImageFolder(root = '/data/driver_detection/test', transform = train_TF)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)

    return train_dataloader, test_dataloader

def divide_subject():
    # divide subject
    subject_list = []
    train_subject_list = []
    test_subject_list = []

    f = open('/data/driver_detection/subject_list.csv','r',encoding='utf-8-sig',newline='')
    rdr = csv.reader(f)
    for line in rdr:
        '''
        line = [subject]
        '''
        subject_list.append(line[0])
    for idx, subject in enumerate(subject_list):
        if idx<int(len(subject_list)*6/7):
            train_subject_list.append(subject)      # 22 train subjects
        else:
            test_subject_list.append(subject)       # 5 train subjects
    del train_subject_list[0]                       # del 'subject' in train subject list

    f.close()
    print('Train Subjects : ', train_subject_list)
    print('Test  Subjects : ', test_subject_list)
    return train_subject_list, test_subject_list

def make_ImageFolder(subject_list, division = 'train'):
    '''
    Make Imagefolder directory
    division_dict = {'safe_drive'          : [],
                     'texting'             : [],
                     'phonecall'           : [],
                     'radio'               : [],
                     'drinking'            : [],
                     'reach_backseat'      : [],
                     'hair_and_makeup'     : [],
                     'talking_to_passenger : []
                    }
    '''
    path = '/data/driver_detection/'

    converting_dict = {'c0':'safe_drive',
                     'c1':'texting',
                     'c2':'phonecall',
                     'c3':'texting',
                     'c4':'phonecall',
                     'c5':'radio',
                     'c6':'drinking',
                     'c7':'reach_backseat',
                     'c8':'hair_and_makeup',
                     'c9':'talking_to_passenger'
                     }
    None_Existing_Labels = ['change_gear','reach_side','standstill_or_waiting']

    division_dict = {}                                  # could be 'train' or 'test'
    for cN in converting_dict:
        division_dict[converting_dict[cN]] = []

    f = open('/data/driver_detection/driver_imgs_list.csv','r',encoding='utf-8-sig',newline='')
    rdr = csv.reader(f)
    for line in rdr:
        '''
        line = [subject,class_name,img_name]
        '''
        subject = line[0]
        c_num = line[1]
        img_name = line[2]

        if subject == 'subject':
            continue
        class_name = converting_dict[c_num]

        if subject in subject_list:
            division_dict[class_name].append(img_name)     # assign file_name according to its class
    f.close()

    # if already dir and data is in the tree, empty it and re-fill
    if os.path.exists(path+division):
        shutil.rmtree(path+division)
    
    for label in division_dict:
        if not os.path.exists(path+'/'+division+'/'+label):
            os.makedirs(path+'/'+division+'/'+label)
        for file in division_dict[label]:
            shutil.copy(path+'imgs/'+file,path+'/'+division+'/'+label)
    
    for non_existing_label in None_Existing_Labels:
        os.makedirs(path+'/'+division+'/'+non_existing_label)