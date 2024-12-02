import random

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch

from src.datasets.anhir import TissueDataset

def seed_worker(worker_id):
    print(f"INITIAL SEED for {worker_id} is {torch.initial_seed()}")
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TissueDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        padded_img_size = config.DATASET.IMG_PAD_SIZE
        # handle train data
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_transform = A.Compose([
            A.LongestMaxSize (max_size=512),
            A.PadIfNeeded(padded_img_size, padded_img_size, position='top_left', border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Affine(scale=(0.8, 1.2),translate_percent=(0, 0.1), rotate=(-30,30),  cval=0, mode=cv2.BORDER_CONSTANT, keep_ratio=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()],
            additional_targets={'target': 'image', 'target_mask': 'mask'})
        # handle val data
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_transform = A.Compose([
            A.LongestMaxSize (max_size=512),
            A.PadIfNeeded(padded_img_size, padded_img_size, position='top_left', border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2()],
            additional_targets={'target': 'image', 'target_mask': 'mask'})
        
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
        }
  
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = TissueDataset([self.train_data_root], transform=self.train_transform, randomly_swap=0.5)
            self.val_dataset = TissueDataset([self.val_data_root], transform=self.val_transform, randomly_swap=0, fraction=1, load_kps=True)
            print(f'Train: {len(self.train_dataset)} images\nValidation: {len(self.val_dataset)} images')
        else:
            raise NotImplementedError('Only fit stage currently supported')
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader_params)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_loader_params)
