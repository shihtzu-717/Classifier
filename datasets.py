# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import pickle
import cv2
import preprocess_data
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from pathlib import Path
from PIL import Image



from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    # crop_img = np.transpose(crop_img, (2,0,1))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def get_split_data(data_root, test_r=0.1, val_r=0.1, file_write=False):
    return preprocess_data.split_data(
        data_root=data_root, 
        test_ratio=test_r, 
        val_ratio=val_r, 
        label_list=['positive', 'negative'], 
        file_write=file_write)


class PotholeDataset(Dataset):
    def __init__(self, data_set, data_path, args, is_train=True, transform=None, target_transform=None):
        super().__init__()
        self.data_set = data_set
        # self.is_train = is_train
        self.data_path = data_path
        self.transform = transform
        self.transform = build_transform(is_train, args)
        self.target_transform = target_transform
       
        self.input_size = args.input_size
        self.padding = args.padding 
        self.padding_size = args.padding_size 
        self.use_shift = args.use_shift 
        self.use_bbox = args.use_bbox 
        self.imsave = args.imsave
        self.upsample = args.upsample
        self.use_class = args.use_class
        
        self.get_crop()

    def __len__(self):
        return self.length

    def get_crop(self):
        img_list=[]
        img_path=[]
        img_bbox=[]
        label_list=[]
        for v in tqdm(self.data_set, desc='Image Cropping... '):
            if v.class_id not in self.use_class:
                continue
            image_path = self.data_path / v.data_set / v.label / v.image_path
            crop_img = preprocess_data.crop_image(
                image_path = image_path, 
                bbox = v.bbox, 
                padding = self.padding, 
                padding_size = self.padding_size, 
                use_shift = self.use_shift, 
                use_bbox = self.use_bbox, 
                imsave = self.imsave
            )
            crop_img = cv2.resize(crop_img, (self.input_size, self.input_size))
            if (crop_img.shape[-1]==3):
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(crop_img)

            # for i in range(self.upsample[v.class_id]):
            img_list.append(pil_image)
            label_list.append(v.label)
            img_path.append(str(image_path))
            img_bbox.append(torch.tensor(v.bbox))

        self.length = len(img_list)
        # ################# set samples ####################
        # import random
        # import shutil
        # samples = random.sample(set(img_path), 200)
        # for sample in samples:
        #     txtsample = sample[:-3] + 'txt'
        #     print(shutil.copy(sample, '/home/daree/code/samples/ss/positive'))
        #     print(shutil.copy(txtsample, '/home/daree/code/samples/ss/positive'))
        # print('done~')
        # ################# set samples ####################

        self.classes = list(np.sort(np.unique(label_list)))
        self.class_to_idx = {string : i for i, string in enumerate(self.classes)}
        self.input_set = (img_list, img_path, img_bbox, label_list) 

    def __getitem__(self, idx):
        image = self.input_set[0][idx]
        img_path = self.input_set[1][idx]
        img_bbox = self.input_set[2][idx]
        label = self.input_set[-1][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        clss = torch.tensor(0) if label == 'negative' else torch.tensor(1)
        return (image, img_path, img_bbox, clss)


if __name__ == "__main__":
    data_root = Path('/home/daree/data/pothole_data/raw')
    sets = preprocess_data.split_data(data_root, 0.1, 0.1, ['positive', 'negative'], file_write=False)
    trainset = PotholeDataset(data_set=sets['train'], args=None, data_path=data_root)
    dataloader = DataLoader(trainset, batch_size=2, shuffle=True)
    print(dataloader)
    for a in dataloader:
        print(a)

