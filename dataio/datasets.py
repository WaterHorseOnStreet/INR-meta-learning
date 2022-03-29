import os
from random import shuffle
import numpy as np

import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from dataio.base import OfficeHomeDataset

batch_size = 64
num_workers = 4

def collate_fn(batch):
    imgs, cls, domain, idx = zip(*batch)

    cls = torch.tensor(cls, dtype=torch.int64)
    domain = torch.tensor(domain, dtype=torch.int64)
    idx = torch.tensor(idx, dtype=torch.int64)

    return torch.stack(imgs, dim=0), cls, domain, idx

def get_HomeOfficeDataloader(dataset_path, source_domain, target_domain):
    train_transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = OfficeHomeDataset(dataset_path, source_domain,train_transforms)
    test_set = OfficeHomeDataset(dataset_path, target_domain, val_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    num_classes = train_set.get_num_classes()
    return train_loader, test_loader, num_classes
