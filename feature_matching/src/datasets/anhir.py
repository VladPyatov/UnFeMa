import os
import random
random.seed(42)

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch.nn.functional as F
import torch


def load_landmarks(landmarks_path):
    landmarks = pd.read_csv(landmarks_path)
    landmarks = landmarks.to_numpy()[:, 1:].astype(np.float32)
    return landmarks

class TissueDataset(Dataset):
    def __init__(self, data_paths, transform=None, randomly_swap=0, fraction=1, load_kps=False, **kwargs):
        self.paths = []
        for data_path in data_paths:
            sample_names = os.listdir(data_path)
            if fraction != 1:
                step = len(sample_names) // int(len(sample_names)*fraction)
                sample_names = sample_names[::step]
            for sample_name in sample_names:
                self.paths.append(os.path.join(data_path, sample_name))
        self.transform = transform
        self.randomly_swap = randomly_swap
        self.load_kps = load_kps
        self.coarse_scale = kwargs.get('coarse_scale', 0.125)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        case_id = self.paths[idx]
        source_path = os.path.join(case_id, "source.mha")
        target_path = os.path.join(case_id, "target.mha")
        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        source_mask =np.ones_like(source)
        target_mask =np.ones_like(target)
        if self.load_kps:
            source_landmarks_path = os.path.join(case_id, "source_landmarks.csv")
            target_landmarks_path = os.path.join(case_id, "target_landmarks.csv")
            source_landmarks = load_landmarks(source_landmarks_path)
            target_landmarks = load_landmarks(target_landmarks_path)
            source_landmarks = torch.from_numpy(source_landmarks)
            target_landmarks = torch.from_numpy(target_landmarks)
            same_points = min(source_landmarks.shape[0], target_landmarks.shape[0])
            source_landmarks = source_landmarks[:same_points]
            target_landmarks = target_landmarks[:same_points]

        if self.transform is not None:
            result = self.transform(image=source, mask=source_mask, target=target, target_mask=target_mask)
            source, target = result['image'], result['target']
            source_mask, target_mask = result['mask'], result['target_mask']
        else:
            source, target = torch.from_numpy(source)[None, None], torch.from_numpy(target)[None, None]
            source_mask, target_mask = torch.from_numpy(source_mask), torch.from_numpy(target_mask)

        if self.coarse_scale:
            [source_mask, target_mask] = F.interpolate(torch.stack([source_mask, target_mask], dim=0)[None].float(),
                                                    scale_factor=self.coarse_scale,
                                                    mode='nearest',
                                                    recompute_scale_factor=False)[0].bool()

        if float(torch.rand(1)) < self.randomly_swap:
            source, target = target, source
            source_mask, target_mask = target_mask, source_mask
            if self.load_kps:
                source_landmarks, target_landmarks = target_landmarks, source_landmarks
        
        data = {'image0': source, 'image1': target, 'mask0': source_mask, 'mask1': target_mask}
        if self.load_kps:
            data.update({'image0_kpts': source_landmarks, 'image1_kpts': target_landmarks})
        return data