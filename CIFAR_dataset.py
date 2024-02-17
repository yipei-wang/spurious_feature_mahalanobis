#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torchvision
import os
import random
import math
import numpy as np

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F


# In[7]:


class CIFAR_subset(Dataset):
    def __init__(self, dataset, indices):
        
        self.data = dataset.data/255.
        self.targets = dataset.targets
        self.indices = indices
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        self.embedding = torch.stack([self.targets, torch.zeros(self.targets.shape)]).T.float()
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.data[idx], self.embedding[idx]
    
class CIFAR_subset_concate(Dataset):
    def __init__(self, dataset, indices, alpha = 0.5, beta = 0.5):
        self.data = torch.FloatTensor(dataset.data/255.)
        self.targets = dataset.targets
        self.indices = indices
        self.data = self.data[indices]
        self.targets = torch.tensor(self.targets)[indices]
        
        self.targets = (self.targets - self.targets.min())/(self.targets.max()-self.targets.min())
        
        self.alpha = alpha
        self.beta = beta
        # in the ordering: y,z_2 = (1,1), (1,0), (0,1), (0,0)
        self.group_size = [
            round(self.alpha*(self.targets == 1).sum().item()),
            (self.targets == 1).sum().item()-round(self.alpha*(self.targets == 1).sum().item()),
            (self.targets == 0).sum().item()-round(self.beta*(self.targets == 0).sum().item()),
            round(self.beta*(self.targets == 0).sum().item()),
        ]
        
        # indicators for each group
        self.grp_11 = np.random.choice(torch.where(self.targets == 1)[0],self.group_size[0],replace = False)
        self.grp_10 = np.array(list(set(torch.where(self.targets == 1)[0].detach().numpy())-set(self.grp_11)))
        self.grp_01 = np.random.choice(torch.where(self.targets == 0)[0],self.group_size[2],replace = False)
        self.grp_00 = np.array(list(set(torch.where(self.targets == 0)[0].detach().numpy())-set(self.grp_01)))
        
        # The part of the blocks that are dependent on the targets
        self.spurious = torch.zeros(self.targets.shape)
        self.spurious[self.grp_01] = 1
        self.spurious[self.grp_11] = 1
        
        self.block = torch.zeros(self.data.shape).float()
        self.grp00_sampled_index = np.random.choice(np.where(self.targets == 0)[0], len(self.grp_00))
        self.grp01_sampled_index = np.random.choice(np.where(self.targets == 1)[0], len(self.grp_01))
        self.grp10_sampled_index = np.random.choice(np.where(self.targets == 0)[0], len(self.grp_10))
        self.grp11_sampled_index = np.random.choice(np.where(self.targets == 1)[0], len(self.grp_11)) 
        
#         print(type(self.data), type(self.block))
#         print(type(self.grp00_sampled_index), type(self.grp_00))
        
        self.block[self.grp_00] = self.data[self.grp00_sampled_index]
        self.block[self.grp_01] = self.data[self.grp01_sampled_index]
        self.block[self.grp_10] = self.data[self.grp10_sampled_index]
        self.block[self.grp_11] = self.data[self.grp11_sampled_index]


        self.data = torch.cat([self.data, self.block], dim = -1)
        self.embedding = torch.stack([self.targets, self.spurious]).T.float()
    
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.data[idx], self.embedding[idx]
    
class CIFAR_subset_watermark(Dataset):
    def __init__(self, dataset, indices, alpha = 0.5, beta = 0.5):
        self.data = torch.FloatTensor(dataset.data/255.)
        self.targets = dataset.targets
        self.indices = indices
        self.data = self.data[indices]
        self.targets = torch.tensor(self.targets)[indices]
        
        self.targets = (self.targets - self.targets.min())/(self.targets.max()-self.targets.min())
        
        self.alpha = alpha
        self.beta = beta
        # in the ordering: y,z_2 = (1,1), (1,0), (0,1), (0,0)
        self.group_size = [
            round(self.alpha*(self.targets == 1).sum().item()),
            (self.targets == 1).sum().item()-round(self.alpha*(self.targets == 1).sum().item()),
            (self.targets == 0).sum().item()-round(self.beta*(self.targets == 0).sum().item()),
            round(self.beta*(self.targets == 0).sum().item()),
        ]
        
        # indicators for each group
        self.grp_11 = np.random.choice(torch.where(self.targets == 1)[0],self.group_size[0],replace = False)
        self.grp_10 = np.array(list(set(torch.where(self.targets == 1)[0].detach().numpy())-set(self.grp_11)))
        self.grp_01 = np.random.choice(torch.where(self.targets == 0)[0],self.group_size[2],replace = False)
        self.grp_00 = np.array(list(set(torch.where(self.targets == 0)[0].detach().numpy())-set(self.grp_01)))
        
        # The part of the blocks that are dependent on the targets
        self.spurious = torch.zeros(self.targets.shape)
        self.spurious[self.grp_01] = 1
        self.spurious[self.grp_11] = 1
        
        self.block = torch.zeros(self.data.shape).float()
        self.grp00_sampled_index = np.random.choice(np.where(self.targets == 0)[0], len(self.grp_00))
        self.grp01_sampled_index = np.random.choice(np.where(self.targets == 1)[0], len(self.grp_01))
        self.grp10_sampled_index = np.random.choice(np.where(self.targets == 0)[0], len(self.grp_10))
        self.grp11_sampled_index = np.random.choice(np.where(self.targets == 1)[0], len(self.grp_11)) 
        
        surrogate = self.data[self.grp_11].clone()
        surrogate[:,28:,28:] = 0
        self.data[self.grp_11] = surrogate.clone()
        surrogate = self.data[self.grp_01].clone()
        surrogate[:,28:,28:] = 0
        self.data[self.grp_01] = surrogate.clone()
    
        self.embedding = torch.stack([self.targets, self.spurious]).T.float()
        
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.data[idx], self.embedding[idx]


# In[8]:


if __name__ == '__main__':
    batch_size = 512
    trainset = torchvision.datasets.CIFAR10('../data', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                 ]))
    testset = torchvision.datasets.CIFAR10('../data', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                 ]))
    
    dig0,dig1 = 0, 8
    where_train = (torch.tensor(trainset.targets) == dig0) + (torch.tensor(trainset.targets) == dig1)
    where_test = (torch.tensor(testset.targets) == dig0) + (torch.tensor(testset.targets) == dig1)
    index_train = torch.where(where_train)[0]
    index_test = torch.where(where_test)[0]
    trainset_concate = CIFAR_subset_concate(trainset, index_train, alpha = .5, beta = .5)
    testset_concate = CIFAR_subset_concate(testset, index_test, alpha = .5, beta = .5)

    trainset_watermark = CIFAR_subset_watermark(trainset, index_train, alpha = .5, beta = .5)
    testset_watermark = CIFAR_subset_watermark(testset, index_test, alpha = .5, beta = .5)


