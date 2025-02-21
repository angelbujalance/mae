import os
import sys
from typing import Any, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

#import transformations
from .augmentations import CropResizing, FTSurrogate, Jitter, Rescaling


class SignalDataset(Dataset):
    """
    Unimodal dataset that generates views of signals.
    """
    def __init__(self, data_path, labels_path=None, labels_mask_path=None, downstream_task=None, 
                 train=False, args=None) -> None:
        """load data and labels from files"""
        self.downstream_task = downstream_task
        
        self.train = train 
        
        self.args = args

        data = torch.load(data_path, map_location=torch.device('cpu')) # load to ram
        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=1)

        self.data = data[..., :self.args.input_electrodes, :]

        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        if self.downstream_task == 'regression':
            data, label, label_mask = self.data[idx], self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd], self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            data, label, label_mask = self.data[idx], self.labels[idx], self.labels_mask[idx]
        
        if self.train == False:
            transform = transforms.Compose([
                CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
        else:
            transform = transforms.Compose([
                CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
                FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                Jitter(sigma=self.args.jitter_sigma),
                Rescaling(sigma=self.args.rescaling_sigma),
                # augmentations.TimeFlip(prob=0.33),
                # augmentations.SignFlip(prob=0.33)
            ])
        data = transform(data)
        
        if self.downstream_task == 'classification':
            label = label.type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        return data, label, label_mask