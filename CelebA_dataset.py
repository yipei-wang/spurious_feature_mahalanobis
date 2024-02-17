import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

class CelebA_HairColor_vs_Gender(Dataset):
    def __init__(
        self,
        data_root = "../data", 
        image_size = 128, 
        train = True, 
        zeta = 0.8,
        seed = 0,
    ):
        self.seed = seed
        self.data_root = data_root
        self.image_size = image_size
        self.train = train
        self.zeta = zeta
        self.transform = transforms.Compose(
            [transforms.Resize((self.image_size,self.image_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                  std = [0.229, 0.224, 0.225])])
        
        self.black_hair = np.loadtxt(os.path.join(data_root, "black_hair.txt"))
        self.male = np.loadtxt(os.path.join(data_root, "male.txt"))
        with open(os.path.join(data_root, "hairColor_gender.txt"), "r") as file:
            self.names = file.readlines()
        self.names = np.array([item[:-1] for item in self.names])
            
        self.group_indices = np.array([
            self.black_hair * self.male,
            self.black_hair * (1-self.male),
            (1-self.black_hair) * self.male,
            (1-self.black_hair) * (1-self.male),
        ])
        if self.train:
            self.indices = np.loadtxt(os.path.join(data_root, f"indices/indices_train_seed{self.seed}.txt")).astype("int")
        else:
            self.indices = np.loadtxt(os.path.join(data_root, f"indices/indices_test_seed{self.seed}.txt")).astype("int")
            
            
        self.names_subset = self.names[self.indices]
        self.group_indices_subset = self.group_indices[:,self.indices]
        self.black_hair_subset = self.black_hair[self.indices]
        self.male_subset = self.male[self.indices]
        self.group_sizes = self.group_indices_subset.sum(1)
        
        if self.train:
            if self.group_sizes[2]/(1-self.zeta) < self.group_sizes[0]:
                self.minor_size = int(self.group_sizes[2])
                self.major_size = int(self.group_sizes[2]/(1-self.zeta)*self.zeta)
            else:
                self.major_size = int(self.group_sizes[0])
                self.minor_size = int(self.group_sizes[0]/self.zeta*(1-self.zeta))
                
            self.indices_balance = np.concatenate([
                np.where(self.group_indices_subset[0])[0][:self.major_size],
                np.where(self.group_indices_subset[1])[0][:self.minor_size],
                np.where(self.group_indices_subset[2])[0][:self.minor_size],
                np.where(self.group_indices_subset[3])[0][:self.major_size]])
            
            self.names_balance = self.names_subset[self.indices_balance]
            self.group_indices_balance = self.group_indices_subset[:,self.indices_balance]
            self.black_hair_balance = self.black_hair_subset[self.indices_balance]
            self.male_balance = self.male_subset[self.indices_balance]
            
            
            
    def __len__(self):
        if self.train:
            return len(self.indices_balance)
        else:
            return len(self.indices)
    
    def __getitem__(self, idx):
        
        if self.train:
            idx = self.indices_balance[idx]
        idx = self.indices[idx]
            
        name = self.names[idx]
        image = Image.open(os.path.join(self.data_root, "img_align_celeba", name))
        label = self.black_hair[idx]
        attr = self.male[idx]
        
        return self.transform(image), label, attr