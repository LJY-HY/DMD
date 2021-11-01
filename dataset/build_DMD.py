import shutil
import os
from torch.utils.data import DataLoader
from util.utils import *
from dataset.ImageFolder import ImageFolder
from PIL import Image

def DMD(args):
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
    train_TF = get_transform('train')
    test_TF = get_transform('test')
    '''
    data shape : [bsz,depth,height,width]
    '''
    divide_subject(test_subject = args.test_subject)
    train_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/train', transform = train_TF)
    val_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/test', transform = test_TF)
    test_dataset = ImageFolder(root = '/data/DMD-Driver-Monitoring-Dataset/test', transform = test_TF)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 16)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 16)
    return train_dataloader, val_dataloader, test_dataloader

def divide_subject(test_subject = 1):
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
    
    if os.path.isfile(path+'check_subject.txt'):
        check_subject = open(path+'check_subject.txt','r')
        subject_num = check_subject.read()
        check_subject.close()
        # if already existing test subject's index == newly introduced subject's index, do nothing
        # if not, clear tree
        if int(subject_num) != test_subject:
            if os.path.exists(path+'train/'):
                shutil.rmtree(path+'train/')
            if os.path.exists(path+'test/'):
                shutil.rmtree(path+'test/')
            check_subject = open(path+'check_subject.txt','w')
            check_subject.write(str(test_subject))
            check_subject.close()
        else:
            return 0
    else:
        check_subject = open(path+'check_subject.txt','w')
        check_subject.write(str(test_subject))
        check_subject.close()

    # copy imgs into the dirs
    build_ImageFolder_shape(train_imgs_list, path, division = 'train')
    build_ImageFolder_shape(test_imgs_list, path, division = 'test')

def divide_video2clip(test_subject = 1):
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
    
    if os.path.isfile(path+'check_subject.txt'):
        check_subject = open(path+'check_subject.txt','r')
        subject_num = check_subject.read()
        check_subject.close()
        if int(subject_num) != test_subject:
            if os.path.exists(path+'train/'):
                shutil.rmtree(path+'train/')
            if os.path.exists(path+'test/'):
                shutil.rmtree(path+'test/')
            check_subject = open(path+'check_subject.txt','w')
            check_subject.write(str(test_subject))
            check_subject.close()
        else:
            return 0
    else:
        check_subject = open(path+'check_subject.txt','w')
        check_subject.write(str(test_subject))
        check_subject.close()

    # copy imgs into the dirs
    build_clip_shape(train_imgs_list, path, division = 'train')
    build_clip_shape(test_imgs_list, path, division = 'test')

def build_ImageFolder_shape(imgs_list, path, division = 'train'):
    for file in imgs_list:
        file_label = file.split('/')[-2]

    # merge labels
        if file_label in ['phonecall_left','phonecall_right']:
            file_label = 'phonecall'
        if file_label in ['texting_left','texting_right']:
            file_label = 'texting'
        if file_label in ['unclassified']:
            continue
        if not os.path.exists(path+'/'+division+'/'+file_label):
            os.makedirs(path+'/'+division+'/'+file_label)
        shutil.copy(file,path+'/'+division+'/'+file_label)

def build_clip_shape(imgs_list, path, division = 'train'):
    '''
    imgs_list를 label과 clip단위로 나누어서 저장
    '''
    # for file in imgs_list:
    #     file_label = file.split('/')[-2]

    # # merge labels
    #     if file_label in ['phonecall_left','phonecall_right']:
    #         file_label = 'phonecall'
    #     if file_label in ['texting_left','texting_right']:
    #         file_label = 'texting'
    #     if file_label in ['unclassified']:
    #         continue
    #     if not os.path.exists(path+'/'+division+'/'+file_label):
    #         os.makedirs(path+'/'+division+'/'+file_label)
    #     shutil.copy(file,path+'/'+division+'/'+file_label)

# class DMD_Video(VisionDataset):
#     def __init__(
#             self,
#             root: str,
#             transform: Optional[Callable] = None,
#     ) -> None:
#         super(DMD_Video, self).__init__(root, transform)
#         self.root = root
#         self.transform = transform

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
#         """
        
#         path = coco.loadImgs(img_id)[0]['file_name']

#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
            
#         return img, target


#     def __len__(self) -> int:
#         return len(self.ids)