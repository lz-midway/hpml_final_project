import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, DistributedSampler

# Dataloader for Food 101 and WikiText
def get_food101_dataloaders(distributed=False,
                            rank=0,
                            world_size=1,
                            data_root = "./data",
                            batch_size = 64,
                            num_workers = 4,
                           prefetch_factor = 4):
    os.makedirs(data_root, exist_ok=True)

    tfms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    tfms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.Food101(root=data_root, split="train",
                                 transform=tfms_train, download=True)
    test_set  = datasets.Food101(root=data_root, split="test",
                                 transform=tfms_test, download=True)
    
    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              sampler=test_sampler, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True)

    return train_loader, test_loader

