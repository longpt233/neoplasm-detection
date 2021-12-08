# for aug 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor

# Loader + pytorch
from torch.utils.data import Dataset, DataLoader
import torch

# other 
import os
import cv2
import numpy as np

# config rgb for mask
mask_values = {
    'neoplastic': [255, 0, 0],
    'non-neoplastic': [0, 255, 0],
}
TRAIN_IMAGE_SIZE = 352 # w*h of image to train


def aug(*, mode='train', train_size=TRAIN_IMAGE_SIZE):
    if mode == 'train':
        return A.Compose([
            A.Resize(train_size, train_size),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                              rotate_limit=10, border_mode=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5), 
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.6),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            
            A.pytorch.ToTensorV2()
        ])
    if mode == 'valid':
        return A.Compose([
            A.Resize(train_size, train_size, p=1.),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class PolypDataset(Dataset):
    def __init__(self, images_id, path2img, path2gt, mask_values=mask_values, train_size=352, transforms=aug(mode='train')):
        super(PolypDataset, self).__init__()
        self.images_id = images_id       
        self.transforms = transforms
        self.train_size = train_size
        self.mask_values = mask_values
        self.path2img = path2img
        self.path2gt = path2gt

    def __len__(self):
        return len(self.images_id)
    def __getitem__(self, idx):
        image_path = os.path.join(self.path2img, f'{self.images_id[idx]}.jpeg')
        image = cv2.imread(image_path)[:,:,::-1]
        img_sz = image.shape
        gt = cv2.imread(os.path.join(self.path2gt, f'{self.images_id[idx]}.jpeg'))[:,:,::-1]
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        neo = np.all(
            gt == self.mask_values["neoplastic"], axis=-1).astype('float')
        non = np.all(
            gt == self.mask_values["non-neoplastic"], axis=-1).astype('float')
        if self.transforms:
            sample = self.transforms(image=image, masks=[neo, non])
            image, masks = sample["image"], sample["masks"]
            neo, non = masks
        
        color = np.full((self.train_size, self.train_size, 2), 0)
        color[:,:,0] = neo
        color[:,:,1] = non
        
        gt = torch.Tensor(color).permute(2, 0, 1)
             
        r = {
            'image': image,
            'gt': gt,
            'id': self.images_id[idx],
            'size': img_sz
        }
        return r


class TestDataset(Dataset):
    
    def __init__(self, images_path, mask_values=mask_values, transforms=aug(mode='valid')):
        super(TestDataset, self).__init__()
        self.images_path = images_path
        self.transforms = transforms
        self.mask_values = mask_values
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        img_id = image_path.split('/')[-1].split('.')[0]
        image = cv2.imread(image_path)[:,:,::-1]
        img_sz = image.shape
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        r = {
            'image': image,
            'id': img_id,
            'size': img_sz
        }
        return r