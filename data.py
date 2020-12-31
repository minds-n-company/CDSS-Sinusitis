# Author : skjang@mnc.ai
# Date : 2020-12-03

import logging
import numpy as np
import pandas as pd
import os, glob, time, datetime
import pickle
import gzip
import copy
import json

import cv2
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from multiprocessing import Pool

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as A
from PIL import Image
from efficientnet_pytorch import EfficientNet

import matplotlib.pyplot as plt
import pydicom as dicom
from pydicom.pixel_data_handlers.numpy_handler import unpack_bits
from torch.utils.tensorboard import SummaryWriter

disease = ['Sinusitis','Oral_cancer'][0]

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

def get_logger(name, level=logging.DEBUG, resetlogfile=False, path='log'):
    fname = os.path.join(path, name+'.log')
    os.makedirs(path, exist_ok=True) 
    if resetlogfile :
        if os.path.exists(fname):
            os.remove(fname) 
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(fname)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

today_datever = datetime.datetime.now().strftime("%y%m%d")    
logger = get_logger(f'{disease}_EfficientNet_{today_datever}', resetlogfile=True)
logger.setLevel(logging.INFO)

def prepare_metatable(filenames):
    cvdf = pd.DataFrame(filenames)
    cvdf.columns = ['filename']
    cvdf['patientID'] = cvdf['filename'].apply(lambda x : x.split('/')[-1].split('_')[0])
    #cvdf['year'] = cvdf['filename'].apply(lambda x : x.split('/')[-2].split('_')[1])

    cvdf['left_label_org'] = cvdf['filename'].apply(lambda x : x.replace('.dcm', '').replace('.jpg', '').split('_')[-2])
    cvdf['right_label_org'] = cvdf['filename'].apply(lambda x : x.replace('.dcm', '').replace('.jpg', '').split('_')[-1])
    #print(pd.crosstab(cvdf['left_label_org'], cvdf['right_label_org'], margins=True))

    cvdf['left_label'] = '1'
    cvdf.at[cvdf['left_label_org']=='0','left_label'] = '0'
    cvdf.at[cvdf['left_label_org']=='x','left_label'] = 'x'
    cvdf['right_label'] = '1'
    cvdf.at[cvdf['right_label_org']=='0','right_label'] = '0'
    cvdf.at[cvdf['right_label_org']=='x','right_label'] = 'x'
    #print(pd.crosstab(cvdf['left_label'], cvdf['right_label']))

    cvdf['FOLD'] = np.nan

    oldcolumns = cvdf.columns.tolist()
    cvdf['index'] = cvdf.index
    cvdf = cvdf[['index']+oldcolumns]
    return cvdf

def save_validationlist(root='.'):
    # list up filenames of valid data
    # totalfiles = glob.glob(os.path.join(root,"test_20??_withUPID","*.dcm"))
    # filenames = glob.glob(os.path.join(root,"test_20??_withUPID","*_[0-3]_[0-3].dcm")) 
    data_dir = ["final_dcm","final_crop"][0]
    logger.info('['*10+' '*20 +'START ANALYSIS'+' '*20+ ']'*10)
    filenames = glob.glob(os.path.join(root,data_dir,"*" + (".dcm" if data_dir=='final_dcm' else '.jpg')))
    logger.info(f'No. of total datasets : {len(filenames)} patients') # 6516

    rmfn = glob.glob(os.path.join(root,data_dir,"*_x_x"+(".dcm" if data_dir=='final_dcm' else '.jpg')))
    if len(rmfn)>1:
        logger.info('  x_x.dcm  :')
        logger.info(rmfn)
        filenames.remove(rmfn)
    logger.info(f'No. of valid datasets : {len(filenames)} patients (excluded x_x.dcm )') #2980 (20.10.7 ver)
    
    cvdf = prepare_metatable(filenames)

    n_folds = 10
    plen = len(filenames)
    logger.info(f'----- Split patients for {n_folds} Cross-validation')

    skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    for ii, (train_pindex, test_pindex) in enumerate(skf.split(range(plen),cvdf['left_label'])):
        # record fold index
        cvdf.at[test_pindex,'FOLD']= ii
        cvdf[f'FOLD{ii}_testset'] = 0
        cvdf.at[test_pindex,f'FOLD{ii}_testset'] = 1

    # save metadata
    filelist_dir = os.path.join(root,'inputlist')
    os.makedirs(filelist_dir, exist_ok=True) 

    cvdf.to_csv(os.path.join(filelist_dir,"input_metadata_table.csv"),index=False)
    cvdf[['index','filename']].to_csv(os.path.join(filelist_dir,"input_filenames_total.csv"),index=False)
    for i in range(n_folds):
        cvdf.loc[cvdf[f'FOLD{i}_testset']==1,'filename'].to_csv(os.path.join(filelist_dir,f"input_filenames_fold{i}.csv"),index=False)

    # statistics
    logger.info(f'----- Data statistics by fold',cvdf['FOLD'].value_counts()) 
    
    logger.info(cvdf['FOLD'].value_counts())

    labelfreq_left = pd.crosstab(cvdf['FOLD'], cvdf['left_label'], margins=True)
    labelfreq_left_ratio = pd.crosstab(cvdf['FOLD'], cvdf['left_label'], margins=True, normalize='index')
    labelfreq_right = pd.crosstab(cvdf['FOLD'], cvdf['right_label'], margins=True)
    labelfreq_right_ratio = pd.crosstab(cvdf['FOLD'], cvdf['right_label'], margins=True, normalize='index')

    labelfreq = pd.concat([labelfreq_left, labelfreq_right], axis=1, keys=['left_sinus', 'right_sinus'], names=[' ','label'])
    labelfreq_ratio = pd.concat([labelfreq_left_ratio, labelfreq_right_ratio], axis=1, keys=['left_sinus', 'right_sinus'], names=[' ','label (ratio)'])
    
    labelfreq.to_csv(os.path.join(filelist_dir,f"label_freq_byfold.csv"))
    labelfreq_ratio.to_csv(os.path.join(filelist_dir,f"label_freq_ratio_byfold.csv"),float_format = '%.2f')
    logger.info(f'----- Label frequency by fold') 
    logger.info(labelfreq)
    logger.info(f'----- Label frequency (ratio) by fold') 
    logger.info(labelfreq_ratio)

import multiprocessing

class ImageDataset(Dataset):
    def __init__(self, root='.', input_csv='inputlist/input_filenames_fold', annotation_path=None, 
    fold_num=0, data_type='train', carrydata=True, transform=None, savejpg=True):
        super(ImageDataset, self).__init__()
        self.root = root
        self.input_csv = input_csv
        self.annotation_path = annotation_path
        self.fold_num = fold_num
        self.data_type = data_type # train, val, test
        self.carrydata = carrydata
        self.transform = transform
        self.savejpg = savejpg

        if self.annotation_path is not None:
            json_file = open(self.annotation_path)
            roi_annotation = json.load(json_file) #coco
            json_file.close()

            self.roi_dict = dict()
            for segmentation in roi_annotation:
                image_name = list(segmentation.keys())[0].replace('.dcm','')  
                bbox_dict = list(segmentation.values())[0][-1]
                assert bbox_dict['name']=='bounding box'
                self.roi_dict[image_name] = bbox_dict['points'] # {'00001334_0_0' : [128.91, 230, 920.48, 786.83]}

        logger.info('--'*20)
        logger.info(f"- Build {self.data_type} dataset") 
        logger.info('-- Transform')
        logger.info(self.transform)
        logger.info('-- ')

        n_folds = 10 

        train_fold = list(range(n_folds))
        val_fold = train_fold[fold_num-1]
        train_fold.remove(fold_num) # test set
        train_fold.remove(val_fold) # validation set

        if data_type=="train":
            self.filenames = []
            for i in train_fold:
                fl_i = pd.read_csv(f'{input_csv}{i}.csv')['filename'].tolist()
                self.filenames.extend(fl_i)
        elif data_type=="val":    
            self.filenames = pd.read_csv(f'{input_csv}{val_fold}.csv')['filename'].tolist()
        elif data_type=="test":    
            self.filenames = pd.read_csv(f'{input_csv}{fold_num}.csv')['filename'].tolist()
        # self.filenames = glob.glob(os.path.join(self.root,"test_20??_withUPID","*_[0-3]_[0-3].dcm")) 

        logger.info(f" Read {len(self.filenames)} files : '{self.filenames[0]}', etc.") # tmp

        self.imagedata = []
        self.targetdata = [] # binary class (0 to 1)
        self.orginal_targetdata = [] # multiclass (0 to 3)

        save_dir = 'preprocessed_data'
        if os.path.exists(save_dir):
            logger.info(f"preprocessed_data folder already exists. Use data in this folder...")
        else:
            logger.info(f"Create preprocessed_data folder...")
            os.makedirs(save_dir, exist_ok=True)


        if self.carrydata: 
            if self.savejpg:
                os.makedirs('./png_tmp', exist_ok=True)

            n_cpu = multiprocessing.cpu_count()
            used_th = 8*6
            logger.info(f'no. cpu existed : {n_cpu}, use {used_th} threads')
            pool = multiprocessing.Pool(processes=used_th)
            result = pool.map(self.read_data, self.filenames)
            pool.close()
            pool.join()

            for i in result :
                self.imagedata.extend(i[1])
                self.orginal_targetdata.extend(i[2])

            self.targetdata = [0 if x==0 else 1 for x in self.orginal_targetdata]
            logger.info(f' No. of datasets loaded  : {len(self.imagedata)} images')


    def __len__(self):
        return len(self.imagedata)

    def __getitem__(self, index):
        if self.carrydata:
            img = self.imagedata[index]
            target = self.targetdata[index]
        else:
            _, img, target = self.read_data(self.filenames[index])
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, target

    def patients_length(self):
        return len(self.filenames)

    def read_data(self, fn, pkl_dir='preprocessed_data'):
        targets = fn.split('.')[-2].split('_')[-2:]
        patient_id = fn.split('/')[-1].split('_')[0]
        
        pkl_fname = os.path.join(pkl_dir, f"{fn.replace('.dcm', '').replace('./', '').replace('/', '__')}.pkl")

        if os.path.exists(pkl_fname):
            # load pickle file
            logger.info(f' > Load preprocessed data : patient # {patient_id}')
            with gzip.open(pkl_fname,'rb') as f:
                [patient_id, imgs, targets] = pickle.load(f)
        else: 
            logger.info(f' > Load & preprocess data : patient # {patient_id}')
            img_format = fn.split('.')[-1]
            if img_format == 'dcm' or img_format == 'DCM' :        
                d = dicom.read_file(fn)
                org_img = d.pixel_array.astype('float')

                logger.info(f' >> min of intensity : {np.min(org_img)}, max: {np.max(org_img)} (patient # {patient_id}, {d.PhotometricInterpretation})')
                
                if d.PhotometricInterpretation == 'MONOCHROME1':
                    org_img = np.max(org_img) - org_img
                
                if self.annotation_path is not None:
                    # crop roi
                    img_name = fn.split('/')[-1].split('.')[0]
                    roi_location = self.roi_dict[img_name] 
                    roi_location = [max(x, 0) for x in roi_location] # for avoiding negative location
                    x1, y1, x2, y2 = roi_location 
                    cropped_img = org_img[int(y1):int(y2),int(x1):int(x2)]

                    plt.imsave(f'png_tmp/{img_name}_original.jpg',org_img, cmap=plt.cm.gray)
                    org_img = cropped_img.copy()

                    os.makedirs('final_crop_fromDicom',exist_ok=True)
                        #cv2.imwrite(f'final_crop_fromDicom/{img_name}.jpg', org_img)
                    plt.imsave(f'final_crop_fromDicom/{img_name}.jpg',cropped_img, cmap=plt.cm.gray)
            elif img_format == 'jpg' or img_format == 'jpeg' or img_format == 'JPG' or \
            img_format == 'png' or img_format == 'PNG':
                org_img = Image.open(fn).convert('L')
                org_img = np.array(org_img)
            if (org_img.shape[0]==0) or (org_img.shape[1]==0):
                logger.warn(f'error in {fn}')
                logger.warn(org_img.shape) #(1174, 1144)
                logger.error(cropped_img.shape) #(574, 0)

            ##tmp : mimic crooped image
            # h, w = org_img.shape
            # h__, w__ = h//10, w//10
            # org_img = org_img[2*h__:8*h__, 2*w__:8*w__] # cropped
            #plt.imsave('./png_tmp/'+patient_id +'-original.png',d.pixel_array, cmap=plt.cm.bone)

            norm_img = self.standardize_image(org_img) ##float64
            imgs = self.split_leftright(norm_img)

            logger.info(f' >> Original image shape : {org_img.shape[0]} x {org_img.shape[1]}') # 1101 x 1134 -> 696 x 636
            logger.info(f' >> Patch shape after split_leftright : {imgs.shape[0]} x {imgs.shape[1]} x {imgs.shape[2]}') # expected : 2 x 318 x 318 # tmp 

            # visualize processed image
            if self.savejpg:
                plt.imsave('./png_tmp/'+patient_id +'-cropped.png', org_img, cmap=plt.cm.bone) # tmp

                if targets[0]!='x':
                    plt.imsave('./png_tmp/'+patient_id +'-left.png',imgs[0,:,:], cmap=plt.cm.bone) # tmp
                if targets[1]!='x':
                    plt.imsave('./png_tmp/'+patient_id +'-right.png',imgs[1,:,:], cmap=plt.cm.bone) # tmp
            
            # drop data if label=='x'
            notx_index = [i for i in range(len(targets)) if targets[i]!='x']
            targets = [targets[i] for i in notx_index]
            targets = [int(x) for x in targets]
            
            imgs = imgs[notx_index,:,:]

            # save pickle file
            objs = [patient_id, imgs, targets]
            with gzip.open(pkl_fname, 'wb') as f:
                pickle.dump(objs, f)
        if 'x' in targets:
                logger.error(fn)
                logger.error(np.where(targets=='x'))
                logger.error('target contains unvalid value')
        return patient_id, imgs, targets

    def split_leftright(self, img_arr):
        def make_square(img_l):
            if img_l.shape[0]>img_l.shape[1]:
                img_l = img_l[1:,:]
            if img_l.shape[1]>img_l.shape[0]:
                img_l = img_l[:,1:]
            return img_l
        h, w = img_arr.shape
        h_, w_ = h//2, w//2
        img_l = img_arr[:, w_:]
        img_r = img_arr[:, :w_]
        img_l = cv2.resize(img_l, (300,300))
        img_r = cv2.resize(img_r, (300,300))
        imgs = np.stack((img_l,img_r),axis=0)
        return imgs

    def image_minmax(self, img):
        img_minmax = img.copy()
        
        if np.ndim(img) == 3:
            for d in range(3):
                img_minmax[d] = image_minmax(img[d])
        else:
            img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
            
        return img_minmax

    def standardize_image(self, img):
        normalized_img = img.copy()
        
        if np.ndim(img) == 3:
            for d in range(3):
                normalized_img[d] = standardize_image(img[d])
        else:
            normalized_img = ((img - np.mean(img)) / np.std(img)).copy()
            
        return normalized_img


def get_dataloaders(args, dataset_class, batch, root, fold_num=0, multinode=False, augment=False):

    logger.info(f'----- Get data loader : Validation No. {fold_num}')

    # augmentation
    transform_test = A.Compose(
        [   
            A.ToTensor()
        ]
    )
    
    if augment:
        transform_train = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                #A.Resize(300,300),
                ToTensorV2()
            ]
        )
    else: 
        transform_train = transform_test

    # prepare training dataset using transform_train
    if args.trained_model is not None:
        train_dataset = None
    else:    
        train_dataset = dataset_class(data_type='train', fold_num=fold_num, transform=transform_train, annotation_path='selected_list(6516).json')

    # prepare validation(for hyperparam. tuning), test sets using transform_test
    if args.trained_model is not None:
        train_dataset = None
    else:    
        val_dataset = dataset_class(data_type='val', fold_num=fold_num, transform=transform_test, annotation_path='selected_list(6516).json')
    test_dataset = dataset_class(data_type='test', fold_num=fold_num, transform=transform_test, annotation_path='selected_list(6516).json')

    train_sampler, valid_sampler, test_sampler = None, None, None 
    trainloader, validloader, testloader = None, None, None 

    #define sampler for imbalanced data
    if args.trained_model is None :

        target_list = torch.tensor(train_dataset.targetdata)
        target_list = target_list[torch.randperm(len(target_list))]
        class_sample_count = np.array([len(np.where(target_list==t)[0]) for t in np.unique(target_list)])
        class_weights = 1./class_sample_count 
        class_weights = torch.from_numpy(class_weights.copy()).type('torch.DoubleTensor')

        logger.info('class_weights : ')
        logger.info(class_weights)

        class_weights_all = class_weights[target_list]

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        # data loader
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch, shuffle=True if train_sampler==None else False, pin_memory=True,
            sampler=train_sampler, drop_last=True)

        validloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch, shuffle=False, pin_memory=True,
            sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch, shuffle=False, pin_memory=True,
        sampler=test_sampler, drop_last=False)

    return train_dataset, trainloader, validloader, testloader
