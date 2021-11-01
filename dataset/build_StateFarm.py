import csv
import os
import shutil
from torch.utils.data import DataLoader
from util.utils import *
from dataset.ImageFolder import ImageFolder

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

    train_subject_list, val_subject_list, test_subject_list = divide_subject(args)
    if train_subject_list != test_subject_list:
        make_ImageFolder(train_subject_list, division = 'train')
        make_ImageFolder(val_subject_list, division='val')
        make_ImageFolder(test_subject_list, division = 'test')
    else:
        make_ImageFolder_deployment(train_subject_list)

    train_dataset = ImageFolder(root = '/data/driver_detection/train', transform = train_TF)
    val_dataset = ImageFolder(root = '/data/driver_detection/val', transform = test_TF)
    test_dataset = ImageFolder(root = '/data/driver_detection/test', transform = test_TF)

    num_workers = 16
    if 'correction' in dir(args):
        if args.correction:
            num_workers = 0
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size*3, shuffle = False, num_workers = 16)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size*3, shuffle = False, num_workers = 16)

    return train_dataloader, val_dataloader, test_dataloader

def divide_subject(args):
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
    train_subject_list = subject_list[1:int(len(subject_list)*5/7)]
    val_subject_list = subject_list[int(len(subject_list)*5/7):int(len(subject_list)*6/7)]
    test_subject_list = subject_list[int(len(subject_list)*6/7):int(len(subject_list))]
    del train_subject_list[0]                       # del 'subject' in train subject list

    f.close()
    if 'deployment_subject' in dir(args):
        if args.deployment_subject != 'all':
            train_subject_list = [args.deployment_subject]
            val_subject_list = [args.deployment_subject]
            test_subject_list = [args.deployment_subject]
        else:
            train_subject_list = subject_list[1:]
            val_subject_list = subject_list[1:]
            test_subject_list = subject_list[1:]

    print('Train      Subjects : ', train_subject_list)
    print('Validation Subjects : ', val_subject_list)
    print('Test       Subjects : ', test_subject_list)
    return train_subject_list, val_subject_list, test_subject_list

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

def make_ImageFolder_deployment(subject_list):
    '''
    Make Imagefolder directory for DEPLOYMENT

    train subject = test subject
    train : val : test = 0.6 : 0.2 : 0.2

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

    for division,ratio_s,ratio_e in zip(['train','val','test'],[0.0,0.6,0.8],[0.6,0.8,1.0]):
        if os.path.exists(path+division):
            shutil.rmtree(path+division)
        for label in division_dict:
            length = len(division_dict[label])
            if not os.path.exists(path+'/'+division+'/'+label):
                os.makedirs(path+'/'+division+'/'+label)
            for file in division_dict[label][int(length*ratio_s):int(length*ratio_e)]:
                shutil.copy(path+'imgs/'+file,path+'/'+division+'/'+label)
        
        for non_existing_label in None_Existing_Labels:
            os.makedirs(path+'/'+division+'/'+non_existing_label)