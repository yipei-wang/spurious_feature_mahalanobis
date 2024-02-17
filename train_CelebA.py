import torch
import torchvision
import os
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from CelebA_dataset import *
from models import *

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:0')


data_root = data_root = "../../../../data_ssd/data/celeba"

## Please select the task: "origin", "regularization", or "groupDRO"
task = "origin"
# task = "regularization"
# task = "groupDRO"

zeta_trains = 1-np.array([0.001, 0.002,0.005,0.01,0.02,0.05,0.1,0.2])
num_epochs = 30
latent_dim = 64








## dataset pre-processing for controllable correlation ratio

# load the attribution labels of all samples
with open(os.path.join(data_root, "Anno", "identity_CelebA.txt"), "r") as f:
    identity = f.readlines()
    identity = np.array([item.split()[0] for item in identity])
with open(os.path.join(data_root, "Anno", "list_attr_celeba.txt"), "r") as f:
    list_attr = f.readlines()
attrs = list_attr[1]
attrs = attrs.split()
attr_labels = []
for i in range(2,len(list_attr)):
    attr_labels.append(list_attr[i].split()[1:])
attr_labels = np.array(attr_labels).astype("int")
attr_labels = (attr_labels+1)/2

# find samples with certain attributions
black_hair = attr_labels[:,attrs.index("Black_Hair")]
blond_hair = attr_labels[:,attrs.index("Blond_Hair")]
male = attr_labels[:,attrs.index("Male")]
female = 1-male

# remove duplicated labeling
valid_indices = (black_hair != blond_hair)
black_hair = black_hair[valid_indices]
blond_hair = blond_hair[valid_indices]
male = male[valid_indices]
female = female[valid_indices]
identity = identity[valid_indices]

Groups = np.array([
    black_hair*male,
    black_hair*female,
    blond_hair*male,
    blond_hair*female
])

# Generate the training/testing subsets for the 10 rounds
if not os.path.exists(os.path.join(data_root, "indices")):
    os.mkdir(os.path.join(data_root, "indices"))
for seed in range(10):
    choices = [
        np.random.RandomState(seed).choice(range(Groups.sum(1)[0].astype("int")),size=(200),replace=False),
        np.random.RandomState(seed).choice(range(Groups.sum(1)[1].astype("int")),size=(200),replace=False),
        np.random.RandomState(seed).choice(range(Groups.sum(1)[2].astype("int")),size=(200),replace=False),
        np.random.RandomState(seed).choice(range(Groups.sum(1)[3].astype("int")),size=(200),replace=False),  
    ]
    indices_test = np.concatenate([
        np.where(Groups[0])[0][choices[0]],
        np.where(Groups[1])[0][choices[1]],
        np.where(Groups[2])[0][choices[2]],
        np.where(Groups[3])[0][choices[3]]
    ]).astype("int")
    indices_train = np.array(list(set(range(len(identity))) - set(indices_test)))
    
    np.savetxt(os.path.join(data_root,f"indices/indices_train_seed{seed}.txt"), indices_train)
    np.savetxt(os.path.join(data_root,f"indices/indices_test_seed{seed}.txt"), indices_test)
    
    
    
    
# Regularization term
def reg(emb, image, label, spurious):

    mu_spur_1 = (emb[(spurious == 1)*(label == 1)].mean(0) + \
                 emb[(spurious == 1)*(label == 0)].mean(0))/2
    mu_spur_0 = (emb[(spurious == 0)*(label == 1)].mean(0) + \
                 emb[(spurious == 0)*(label == 0)].mean(0))/2

    mu_spur = ((mu_spur_1-mu_spur_0)/2).reshape(-1,1)
    sig_spur = ((mu_spur_1 - emb[spurious == 1]).T.mm(
        (mu_spur_1 - emb[spurious == 1])) + \
                    (mu_spur_0 - emb[spurious == 0]).T.mm(
        (mu_spur_0 - emb[spurious == 0])))/(len(trainset) - 1)

    mu_info_1 = (emb[(spurious == 1)*(label == 1)].mean(0) + \
                 emb[(spurious == 0)*(label == 1)].mean(0))/2
    mu_info_0 = (emb[(spurious == 1)*(label == 0)].mean(0) + \
                 emb[(spurious == 0)*(label == 0)].mean(0))/2

    mu_info = ((mu_info_1-mu_info_0)/2).reshape(-1,1)

    loss = mu_spur.norm()/mu_info.norm()
    return loss




if task == "origin":
    for seed in range(10):
        print(f'=========================={seed}=============================')
        for model_name in ['resnet', 'vgg', 'alexnet']:
            if model_name == 'resnet':
                batch_size = 512
            else:
                batch_size = 256
            print(f'========================{model_name}==========================')
            for zeta_train in zeta_trains:
                print(f'========================{zeta_train}===========================')
                torch.manual_seed(seed)
                np.random.seed(seed)

                trainset = CelebA_HairColor_vs_Gender(data_root=data_root, train=True, zeta = zeta_train, seed = seed)
                testset = CelebA_HairColor_vs_Gender(data_root=data_root, train=False, seed = seed)
                trainloader = DataLoader(trainset, batch_size=batch_size, num_workers = 8, shuffle=True)
                testloader = DataLoader(testset, batch_size=batch_size, num_workers = 8, shuffle=False)

                Loss = []
                if model_name == 'resnet':
                    model = Classifier_resnet(latent_dim).to(device)
                elif model_name == 'vgg':
                    model = Classifier_vgg(latent_dim).to(device)
                elif model_name == 'alexnet':
                    model = Classifier_alexnet(latent_dim).to(device)

                optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=.9,weight_decay=5e-5)
                Acc_train = []
                Acc_test = []
                Cls_loss = []
                Reg_loss = []
                criterion = nn.BCELoss()
                model.train()
                for epoch in range(num_epochs):
                    cls_loss = 0.0
                    reg_loss = 0.0
                    acc_train = 0
                    model.train()
                    for data in trainloader:
                        images, label, spurious = data
                        images = images.to(device)
                        label = label.unsqueeze(1).to(device)
                        spurious = spurious.unsqueeze(1).to(device)
                        optimizer.zero_grad()
                        emb, pred = model(images)

                        with torch.no_grad():
                            if ((label.squeeze() == 1)*(spurious.squeeze() == 0)).sum().item()*((label.squeeze() == 0)*(spurious.squeeze() == 1)).sum().item() == 0:
                                loss_reg = 0
                            else:
                                loss_reg = reg(emb,images,label.squeeze(), spurious.squeeze())

                        loss_cls = criterion(pred, label.float())
                        loss = loss_cls
                        loss.backward()
                        optimizer.step()
                        cls_loss += loss_cls.item()
                        if loss_reg > 0:
                            reg_loss += loss_reg.item()
                        acc_train += ((pred > 0.5)==label).float().sum().item()

                    acc_train/=len(trainset)

                    acc_test = 0
                    model.eval()
                    with torch.no_grad():
                        for data in testloader:
                            images, label, spurious = data
                            images = images.to(device)
                            label = label.unsqueeze(1).to(device)
                            spurious = spurious.unsqueeze(1).to(device)
                            optimizer.zero_grad()
                            _, pred = model(images)
                            acc_test += ((pred > 0.5)==label).float().sum().item()
                    acc_test/=len(testset)


                    print('S/M/Z: [%d/%s/%.3f] Epoch [%d/%d], Loss: [%.4f,%.4f], Acc: [%.4f/%.4f]' % (
                        seed, model_name, zeta_train, epoch+1, num_epochs, cls_loss/len(trainloader),
                        reg_loss/len(trainloader), acc_train, acc_test))

                    Acc_test.append(acc_test)
                    Acc_train.append(acc_train)
                    Cls_loss.append(cls_loss/len(trainloader))
                    Reg_loss.append(reg_loss/len(trainloader))

                torch.save(model.state_dict(), f'results/CelebA/noReg_{model_name}_{zeta_train}_{num_epochs}_seed{seed}')
                np.savetxt(f'results/CelebA/trainAcc_{model_name}_{zeta_train}_noReg_{num_epochs}_seed{seed}.txt', Acc_train)
                np.savetxt(f'results/CelebA/testAcc_{model_name}_{zeta_train}_noReg_{num_epochs}_seed{seed}.txt', Acc_test)
                np.savetxt(f'results/CelebA/clsLoss_{model_name}_{zeta_train}_noReg_{num_epochs}_seed{seed}.txt', Cls_loss)
                np.savetxt(f'results/CelebA/regLoss_{model_name}_{zeta_train}_noReg_{num_epochs}_seed{seed}.txt', Reg_loss)
                
                
                
                
                
                
if task == "regularization":
    for seed in range(10):
        print(f'=========================={seed}=============================')
        for model_name in ['resnet']:
            if model_name == 'resnet':
                batch_size = 512
            else:
                batch_size = 256
            print(f'========================{model_name}==========================')
            for zeta_train in zeta_trains:
                print(f'========================{zeta_train}===========================')
                torch.manual_seed(seed)
                np.random.seed(seed)

                trainset = CelebA_HairColor_vs_Gender(data_root = data_root, train=True, zeta = zeta_train, seed = seed)
                testset = CelebA_HairColor_vs_Gender(data_root = data_root, train=False, seed = seed)
                trainloader = DataLoader(trainset, batch_size=batch_size, num_workers = 8, shuffle=True)
                testloader = DataLoader(testset, batch_size=batch_size, num_workers = 8, shuffle=False)

                Loss = []
                if model_name == 'resnet':
                    model = Classifier_resnet(latent_dim).to(device)
                elif model_name == 'vgg':
                    model = Classifier_vgg(latent_dim).to(device)
                elif model_name == 'alexnet':
                    model = Classifier_alexnet(latent_dim).to(device)

                optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=.9,weight_decay=5e-5)
                Acc_train = []
                Acc_test = []
                Cls_loss = []
                Reg_loss = []
                criterion = nn.BCELoss()
                model.train()
                for epoch in range(num_epochs):
                    cls_loss = 0.0
                    reg_loss = 0.0
                    acc_train = 0
                    model.train()
                    for data in trainloader:
                        images, label, spurious = data
                        images = images.to(device)
                        label = label.unsqueeze(1).to(device)
                        spurious = spurious.unsqueeze(1).to(device)
                        optimizer.zero_grad()
                        emb, pred = model(images)

                        if ((label.squeeze() == 1)*(spurious.squeeze() == 0)).sum().item()*((label.squeeze() == 0)*(spurious.squeeze() == 1)).sum().item() == 0:
                            loss_reg = 0
                        else:
                            loss_reg = reg(emb,images,label.squeeze(), spurious.squeeze())

                        loss_cls = criterion(pred, label.float())
                        loss = loss_cls + loss_reg
                        loss.backward()
                        optimizer.step()
                        cls_loss += loss_cls.item()
                        if loss_reg > 0:
                            reg_loss += loss_reg.item()
                        acc_train += ((pred > 0.5)==label).float().sum().item()

                    acc_train/=len(trainset)

                    acc_test = 0
                    model.eval()
                    with torch.no_grad():
                        for data in testloader:
                            images, label, spurious = data
                            images = images.to(device)
                            label = label.unsqueeze(1).to(device)
                            spurious = spurious.unsqueeze(1).to(device)
                            optimizer.zero_grad()
                            _, pred = model(images)
                            acc_test += ((pred > 0.5)==label).float().sum().item()
                    acc_test/=len(testset)


                    print('S/M/Z: [%d/%s/%.3f] Epoch [%d/%d], Loss: [%.4f,%.4f], Acc: [%.4f/%.4f]' % (
                        seed, model_name, zeta_train, epoch+1, num_epochs, cls_loss/len(trainloader),
                        reg_loss/len(trainloader), acc_train, acc_test))

                    Acc_test.append(acc_test)
                    Acc_train.append(acc_train)
                    Cls_loss.append(cls_loss/len(trainloader))
                    Reg_loss.append(reg_loss/len(trainloader))

            #         lr_scheduler.step()


                torch.save(model.state_dict(), f'results/CelebA/Reg_{model_name}_{zeta_train}_{num_epochs}_seed{seed}')
                np.savetxt(f'results/CelebA/trainAcc_{model_name}_{zeta_train}_Reg_{num_epochs}_seed{seed}.txt', Acc_train)
                np.savetxt(f'results/CelebA/testAcc_{model_name}_{zeta_train}_Reg_{num_epochs}_seed{seed}.txt', Acc_test)
                np.savetxt(f'results/CelebA/clsLoss_{model_name}_{zeta_train}_Reg_{num_epochs}_seed{seed}.txt', Cls_loss)
                np.savetxt(f'results/CelebA/regLoss_{model_name}_{zeta_train}_Reg_{num_epochs}_seed{seed}.txt', Reg_loss)
                
                
                
                
                
                
                
if task == "groupDRO":
    criterion = nn.BCELoss(reduction="none")
    def group_accuracy(model, trainSubloader):
        model.eval()
        with torch.no_grad():
            Acc = []
            for subloader in trainSubloader:
                acc = 0
                ct = 0
                for data in subloader:
                    images, label = data
                    images = images.permute(0,3,1,2).to(device)
                    label = label[:,0].unsqueeze(1).to(device)
                    _, pred = model(images)
                    acc += ((pred > 0.5)==label).float().sum().item()
                    ct += len(images)
    #             print(acc/ct)
                Acc.append(acc/ct)
        return Acc

    def total_accuracy(model, dataloader, group = False):
        model.eval()
        if not group: 
            acc = 0
            ct = 0
        else:
            acc = [0,0,0,0]
            ct = [0,0,0,0]
        for data in dataloader:
            images, label, spurious = data
            images = images.to(device)
            spurious = spurious.to(device)
            label = label.unsqueeze(1).to(device)
            _, pred = model(images)

            if not group:
                acc += ((pred > 0.5)==label[None]).float().sum().item()
                ct += len(images)
            else:
                grp = torch.tensor([group_mapping[(label[i].item(), spurious[i].item())] for i in range(len(label))])
                grp = grp.to(device)
                for i in range(4):
                    acc[i] += ((pred > 0.5)==label)[grp == i].float().sum().item()
                    ct[i] += (grp == i).float().sum().item()

        if not group:      
            return acc/ct
        else:
            return [acc[0]/ct[0],acc[1]/ct[1],acc[2]/ct[2],acc[3]/ct[3]]


    def train(model, dataloader):
        model.train()
        cls_loss = 0
        reg_loss = 0
        for idx, data in enumerate(dataloader):
            images, label, spurious = data
            images = images.to(device)
            spurious = spurious.to(device)
            label = label.unsqueeze(1).to(device).float()
            grp = torch.tensor([group_mapping[(label[i].item(), spurious[i].item())] for i in range(len(label))])
            grp = grp.to(device)

            emb, pred = model(images)
            loss_cls = loss_fn(pred, label, grp)
    #         loss_cls = criterion(pred, label).mean()

            with torch.no_grad():
                if ((spurious.flatten() == 1)*(label.flatten() == 0)).sum().item()*((spurious.flatten() == 0)*(label.flatten() == 1)).sum().item() == 0:
                    loss_reg = 0
                else:
                    loss_reg = reg(emb,images, label.flatten(), spurious.flatten())

            loss = loss_cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cls_loss += loss_cls.item()
            if loss_reg>0:
                reg_loss += loss_reg.item()
        return cls_loss, reg_loss

    group_mapping = {
        (1,1): 0, (1,0): 1, (0,1): 2, (0,0):3
    }

    def compute_grp_avg(losses, grp):
        grp_map = (grp == torch.arange(4).unsqueeze(1).long().to(losses.device)).float()
        grp_count = grp_map.sum(1)
        grp_denom = grp_count + (grp_count==0).float() # avoid nans
    #     print(losses)
        grp_loss = (grp_map @ losses.view(-1))/grp_denom
        return grp_loss, grp_count

    def compute_robust_loss(grp_loss, grp_count, step_size = 1e-2):
        adv_probs = torch.ones(4).to(grp_loss.device)/4
        adjusted_loss = grp_loss
        adv_probs = torch.exp(step_size*adjusted_loss.data)
        adv_probs = adv_probs/(adv_probs.sum())
        robust_loss = grp_loss @ adv_probs
        return robust_loss, adv_probs

    def loss_fn(pred, label, grp):
        per_sample_losses = criterion(pred, label)
        grp_loss, grp_count = compute_grp_avg(per_sample_losses, grp)
        grp_acc, grp_count = compute_grp_avg(((pred > 0.5) == label).float(), grp)
        actual_loss, weights = compute_robust_loss(grp_loss, grp_count)
        return actual_loss

    for seed in range(10):
        print(f'=========================={seed}=============================')
        for model_name in ['resnet']:
            if model_name == 'resnet':
                batch_size = 512
            else:
                batch_size = 256
            print(f'========================{model_name}==========================')
            for zeta_train in zeta_trains:
                print(f'========================{zeta_train}===========================')
                torch.manual_seed(seed)
                np.random.seed(seed)

                trainset = CelebA_HairColor_vs_Gender(data_root, train=True, zeta = zeta_train, seed = seed)
                testset = CelebA_HairColor_vs_Gender(data_root, train=False, seed = seed)
                trainloader = DataLoader(trainset, batch_size=batch_size, num_workers = 8, shuffle=True)
                testloader = DataLoader(testset, batch_size=batch_size, num_workers = 8, shuffle=False)

                Loss = []
                if model_name == 'resnet':
                    model = Classifier_resnet(latent_dim).to(device)
                elif model_name == 'vgg':
                    model = Classifier_vgg(latent_dim).to(device)
                elif model_name == 'alexnet':
                    model = Classifier_alexnet(latent_dim).to(device)

                optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,'min',factor=0.1,patience=5,threshold=0.0001,min_lr=0,eps=1e-08)
                Acc_train = []
                Acc_test = []
                Cls_loss = []
                Reg_loss = []
                criterion = nn.BCELoss(reduction='none')
                for epoch in range(num_epochs):

                    cls_loss, reg_loss = train(model, trainloader)
                    Acc = total_accuracy(model, trainloader, True)
                    scheduler.step(cls_loss)
                    worst_group = Acc.index(min(Acc))
                    acc_test = total_accuracy(model, testloader)

                    print('Epoch [%d/%d], grp: [%d], Loss: [%.4f/%.4f], Group Acc: [%.3f, %.3f, %.3f, %.3f] Test: [%.4f]' % (
                        epoch+1, num_epochs, worst_group, cls_loss/len(trainloader), reg_loss/len(trainloader),
                        Acc[0], Acc[1], Acc[2], Acc[3], acc_test))

                    Acc_train.append(Acc)
                    Acc_test.append(acc_test)
                    Cls_loss.append(cls_loss/len(trainloader))
                    Reg_loss.append(reg_loss/len(trainloader))


                torch.save(model.state_dict(), f'results/CelebA/GroupDRO/DRO_{model_name}_{zeta_train}_{num_epochs}_seed{seed}')
                np.savetxt(f'results/CelebA/trainAcc_{model_name}_{zeta_train}_DRO_{num_epochs}_seed{seed}.txt', Acc_train)
                np.savetxt(f'results/CelebA/testAcc_{model_name}_{zeta_train}_DRO_{num_epochs}_seed{seed}.txt', Acc_test)
                np.savetxt(f'results/CelebA/clsLoss_{model_name}_{zeta_train}_DRO_{num_epochs}_seed{seed}.txt', Cls_loss)
                np.savetxt(f'results/CelebA/regLoss_{model_name}_{zeta_train}_DRO_{num_epochs}_seed{seed}.txt', Reg_loss)


