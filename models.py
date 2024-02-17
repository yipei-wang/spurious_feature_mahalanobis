import torch
import torchvision
from torch import nn


class Classifier_resnet(nn.Module):
    def __init__(self, latent_dim = 128):
        super(Classifier_resnet, self).__init__()
        
        # cnn
        self.resnet = torchvision.models.resnet18(pretrained=True)
        
        self.emb = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        f = self.emb(torch.flatten(x, 1))
        x = self.fc(f)
        return f,x

class Classifier_vgg(nn.Module):
    def __init__(self, latent_dim = 128):
        super(Classifier_vgg, self).__init__()
        
        # cnn
        self.vgg = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        
        
        self.emb = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg(x)
        
        f = self.emb(torch.flatten(x, 1))
        x = self.fc(f)
        return f,x

class Classifier_alexnet(nn.Module):
    def __init__(self, latent_dim = 128):
        super(Classifier_alexnet, self).__init__()
        
        # cnn
        self.alexnet = torchvision.models.alexnet(pretrained=True).features[:11]
        
        self.emb = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.alexnet(x)
        f = self.emb(torch.flatten(x, 1))
        x = self.fc(f)
        return f,x