#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torchvision
import os
import random
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F


# In[7]:


class water_land_bird(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_root, tensor_image = False, image_size = 128, train = False):
        """
        Args:
            data_root (string): Path to the data.
            CUB_transform (callable): transform to be applied on all data.
            tensor_image (bool): whether or not to transform the images to tensors.
        """
        self.data_root = data_root
        self.tensor_image = tensor_image
        self.image_size = image_size
        self.max_object_size = 0.15
        self.max_rotation_angle = 30
        self.train = train
        
        if self.tensor_image:
            self.image_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
#                  transforms.Normalize(mean = [0.485, 0.456, 0.406], 
#                                       std = [0.229, 0.224, 0.225])
                ])
            self.mask_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
                ])
        else:
            self.image_transform = None
            
        # train test split: a binary list of the is_trainset of 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/train_test_split.txt'), 'r')  as f:
            self.train_test_split = f.readlines()
        self.train_test_split = np.array([int(att.split()[1]) for att in self.train_test_split])
        
        if self.train:
            self.indices = np.where(self.train_test_split == 1)[0]
        else:
            self.indices = np.where(self.train_test_split == 0)[0]
        
        # image list: a list of 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/images.txt'), 'r')  as f:
            self.image_list = f.readlines()
        self.image_list = np.array([att.split()[1] for att in self.image_list])
        self.image_list = self.image_list[self.indices]
        
        
        # class list: a list of 200 classes
        with open(os.path.join(self.data_root, './CUB_200_2011/classes.txt'), 'r')  as f:
            self.class_list = f.readlines()
        self.class_list = [att.split()[1] for att in self.class_list]
        
        # image to label: a list of the classes of the 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/image_class_labels.txt'), 'r')  as f:
            self.image2label = f.readlines()
        self.image2label = np.array([int(att.split()[1])-1 for att in self.image2label])
        self.image2label = self.image2label[self.indices]
               
        self.water_birds_list = [
            'Albatross', # Seabirds
            'Auklet',
            'Cormorant',
            'Frigatebird',
            'Fulmar',
            'Gull',
            'Jaeger',
            'Kittiwake',
            'Pelican',
            'Puffin',
            'Tern',
            'Gadwall', # Waterfowl
            'Grebe',
            'Mallard',
            'Merganser',
            'Guillemot',
            'Pacific_Loon'
        ]
        
        # get the label for the water/land birds
        self.unresampled_label = []
        for i in range(len(self.image_list)):
            is_waterbird = False
            for waterbird in self.water_birds_list:
                if waterbird in self.image_list[i]:
                    is_waterbird = True
                    break
            if is_waterbird:
                self.unresampled_label.append(1)
            else:
                self.unresampled_label.append(0)
        self.unresampled_label = np.array(self.unresampled_label)
        
        self.waterbird_resampled_indices = np.random.RandomState(seed=42).choice(
            np.where(self.unresampled_label == 1)[0],
            size = ((self.unresampled_label == 0).sum() - (self.unresampled_label == 1).sum()))
        self.waterbird_indices = np.concatenate(
            [np.where((self.unresampled_label == 1))[0], self.waterbird_resampled_indices])  
        
        self.landbird_indices = np.where((self.unresampled_label == 0))[0]
        
        self.total_indices = np.concatenate([
            self.waterbird_indices, self.landbird_indices
        ])
        
    
    def resize(self, x, mask, ratio):
        if torch.is_tensor(ratio):
            ratio = ratio.item()
        pad_size = x.shape[-1] - int(x.shape[-1]*np.sqrt(ratio))
        new_x = F.interpolate(x[None], scale_factor = np.sqrt(ratio), mode = 'bilinear', align_corners = False)
        x = F.pad(new_x, (pad_size//2, pad_size-pad_size//2)*2,
                  mode = 'constant', value = 0).squeeze()
    
        new_mask = F.interpolate(mask[None], scale_factor = np.sqrt(ratio), mode = 'bilinear', align_corners = False)
        mask = F.pad(new_mask, (pad_size//2, pad_size-pad_size//2)*2,
                  mode = 'constant', value = 0).squeeze()
        return x, mask
    
    def rotate(self, x, mask):
        angle = (np.random.rand()*2-1)*self.max_rotation_angle
        x_rot = torchvision.transforms.functional.rotate(x[None], angle).squeeze()
        mask_rot = torchvision.transforms.functional.rotate(mask[None], angle).view(1,self.image_size,self.image_size)
        return x_rot, mask_rot
        
    def __len__(self):
        return len(self.total_indices)

    def __getitem__(self, idx):
        
        idx = self.total_indices[idx]
        image = Image.open(os.path.join(self.data_root, './CUB_200_2011/images', self.image_list[idx]))
        mask = Image.open(os.path.join(self.data_root, 'segmentations', 
                                       self.image_list[idx].split('.jpg')[0]+'.png'))
        image = image.convert('RGB')
        label = self.unresampled_label[idx]
        
        if self.tensor_image:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)[0][None]
            masked = image*mask
            mask_size = (mask > 0).float().mean()
            if mask_size > self.max_object_size:
                masked, mask = self.resize(masked, mask, self.max_object_size/mask_size)
            masked, mask = self.rotate(masked, mask)
            return masked, mask, label
        else:
            return image, mask, label
        
        
class places(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_root, image_size = 128, tensor_image = True, train = False):
        """
        Args:
            data_root (string): Path to the data.
            CUB_transform (callable): transform to be applied on all data.
            tensor_image (bool): whether or not to transform the images to tensors.
        """
        self.data_root = data_root
        self.tensor_image = tensor_image
        self.train = train
        self.image_size = image_size
        
        if self.tensor_image:
            self.image_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
#                  transforms.Normalize(mean = [0.485, 0.456, 0.406], 
#                                       std = [0.229, 0.224, 0.225])
#                  transforms.Normalize(mean = [0., 0., 0.], 
#                                       std = [1., 1., 1.])
                ])
        else:
            self.image_transform = None
            
        self.land = ['b/bamboo_forest', 'f/forest/broadleaf']
        self.water = ['l/lake/natural', 'o/ocean']
        
        self.image_list = []
        self.label = []
        
        # training images are the first 2500 of all four categories, totally 10000
        # testing images are the last 2500 of all four catetories, totally 10000
        if self.train:
            for land_class in self.land:
                self.image_list += [os.path.join(self.data_root, land_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500)]
            for water_class in self.water:
                self.image_list += [os.path.join(self.data_root, water_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500)]
        else:
            for land_class in self.land:
                self.image_list += [os.path.join(self.data_root, land_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500,5000)]
            for water_class in self.water:
                self.image_list += [os.path.join(self.data_root, water_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500,5000)]
        self.image_list = np.array(self.image_list)
        self.label = np.array([0]*5000+[1]*5000)
        
        
        
    def __len__(self):
        return len(self.total_indices)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_list[idx])
        
        image = image.convert('RGB')
        label = self.label[idx]
        return self.image_transform(image), label
    
    
    
class SpuriousBirds(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, CUB_root, place_root, image_size=128, train=False, alpha=0.5, beta=0.5):

        self.train=train
        self.CUB_root=CUB_root
        self.place_root=place_root
        self.image_size=image_size
        self.alpha, self.beta = alpha, beta
        
        self.set_bird=water_land_bird(self.CUB_root,tensor_image=True,image_size=self.image_size,train=self.train)
        self.set_place=places(self.place_root,tensor_image=True,image_size=self.image_size,train=self.train)
        
        self.bird_label=self.set_bird.unresampled_label[self.set_bird.total_indices]
        self.place_label=self.set_place.label

        self.group_size = [
            round(self.alpha*len(self.set_bird)/2),
            len(self.set_bird)//2-round(self.alpha*len(self.set_bird)/2),
            len(self.set_bird)//2-round(self.beta*len(self.set_bird)/2),
            round(self.beta*len(self.set_bird)/2),
        ]
        self.spurious = [1]*self.group_size[0]+[0]*self.group_size[1]+[1]*self.group_size[2]+[0]*self.group_size[3]
        self.spurious = np.array(self.spurious)
        self.background_for_waterbird = np.array([
            5000+i for i in range(self.group_size[0])]+[i for i in range(self.group_size[1])])
        self.background_for_landbird = np.array([
            5000+i for i in range(self.group_size[2])]+[i for i in range(self.group_size[3])])
        self.place_permute = np.concatenate([self.background_for_waterbird,
                                             self.background_for_landbird])
        self.generate_images()
        
    def __len__(self):
        return len(self.set_bird)
    
    def generate_images(self):
        self.image = []
        for idx in tqdm(range(len(self.set_bird)//2)):
            bird, mask, bird_label = self.set_bird[idx]
            place, place_label = self.set_place[self.place_permute[idx]]
            self.image.append(place*(1-mask)+bird)
        for idx in tqdm(range(len(self.set_bird)//2,len(self.set_bird))):
            bird, mask, bird_label = self.set_bird[idx]
            place, place_label = self.set_place[self.place_permute[idx]]
            self.image.append(place*(1-mask)+bird)
        self.image = torch.stack(self.image)

    def __getitem__(self, idx):        
        return self.image[idx], self.bird_label[idx], self.spurious[idx]  
    


if __name__ == '__main__':
    batch_size = 256
    trainset = SpuriousBirds(CUB_root, place_root, train = True,alpha=zeta_train,beta=zeta_train)
    testset = SpuriousBirds(CUB_root, place_root, train = False)
    
    fig,ax = plt.subplots(1,4,figsize=(10,3))

    ax[0].imshow(trainset.image[0].permute(1,2,0))
    ax[1].imshow(trainset.image[4000].permute(1,2,0))
    ax[2].imshow(trainset.image[5000].permute(1,2,0))
    ax[3].imshow(trainset.image[8000].permute(1,2,0))
    ax[0].set_title('waterbird, water')
    ax[1].set_title('waterbird, land')
    ax[2].set_title('landbird, water')
    ax[3].set_title('landbird, land')
    for i in range(4):
        ax[i].axis('off')
    plt.tight_layout()
#     plt.savefig('waterbird-demonstration.pdf', bbox_inches = 'tight')
    plt.show()


