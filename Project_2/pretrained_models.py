# -*- coding: utf-8 -*-
"""pretrained_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mfZh3MSjP2T0uzlPsWq2X1vZTz-tqZmS

# Importing libraries
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.optim import SGD, Adam, lr_scheduler
from copy import deepcopy
from utils import *
from tta import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([                                      
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

"""# Pretrained Resnet, changing only last layer, finetuning all layers"""

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features = in_features, out_features = 10)
    
    def forward(self,x):
        return self.pretrained_model(x)

model = ResnetCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
train_loader, val_loader, test_loader = load_data(train_transform, test_transform)

train_network(model, train_loader, val_loader, epochs = 5, print_results=False, 
              criterion = criterion, optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

"""# Pretrained Resnet, changing only last layer, finetuning all layers, dropout before last layer"""

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
                                  nn.Dropout(),
                                  nn.Linear(in_features, 10)
                              )
    
    def forward(self,x):
        return self.pretrained_model(x)

model = ResnetCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
train_network(model, train_loader, val_loader, epochs = 5, print_results=False, 
              criterion = criterion, optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

"""# Pretrained Resnet, changing only last layer, finetuning last layer only"""

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, 10)
    
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        for param in self.pretrained_model.fc.parameters():
            param.requires_grad = True

    def forward(self,x):
        return self.pretrained_model(x)

model = ResnetCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
train_network(model, train_loader, val_loader, epochs = 5, print_results=False, 
              criterion = criterion, optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

"""# Pretrained Resnet, changing only last layer, finetuning last few layers"""

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, 10)
    
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        for param in self.pretrained_model.fc.parameters():
            param.requires_grad = True

        for param in self.pretrained_model.layer4.parameters():
            param.requires_grad = True

    def forward(self,x):
        return self.pretrained_model(x)

model = ResnetCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
train_network(model, train_loader, val_loader, epochs = 5, print_results=False, 
              criterion = criterion, optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

"""# Different optimizers"""

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, 10)
    
    def forward(self,x):
        return self.pretrained_model(x)

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)

model = ResnetCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_network(model, train_loader, val_loader, epochs = 5, print_results=True, 
            optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

"""# Different augmentations on training set"""

train_transform = transforms.Compose([                                      
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, 10)
    
    def forward(self,x):
        return self.pretrained_model(x)

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)

model = ResnetCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_network(model, train_loader, val_loader, epochs = 5, print_results=True, 
            optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

train_transform = transforms.Compose([                                      
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, 10)
    
    def forward(self,x):
        return self.pretrained_model(x)

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)

model = ResnetCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_network(model, train_loader, val_loader, epochs = 5, print_results=True, 
            optimizer = optimizer, scheduler=scheduler)
print('Accuracy on validation set: ', evaluate_network(model, val_loader))
print('Accuracy on test set: ', evaluate_network(model, test_loader))
plot_loss(model)
plot_accuracy(model)

"""# Test time Augmentations"""

tta = [transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
])]

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
print('TTA: HorizontalFlip')
print('validation with tta: ', evaluate_tta(model, val_loader, tta, 5000))
print('test with tta: ', evaluate_tta(model, test_loader, tta, 10000))
print('-'*50)
print()


tta = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
]), 
    transforms.Compose([
        transforms.RandomRotation(10),
])
]

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
print('TTA: HorizontalFlip and Rotation')
print('validation with tta: ', evaluate_tta(model, val_loader, tta, 5000))
print('test with tta: ', evaluate_tta(model, test_loader, tta, 10000))
print('-'*50)
print()


tta = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
]), 
    transforms.Compose([
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
])
]

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
print('TTA: HorizontalFlip and crop')
print('validation with tta: ', evaluate_tta(model, val_loader, tta, 5000))
print('test with tta: ', evaluate_tta(model, test_loader, tta, 10000))
print('-'*50)
print()



tta = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
]), 
    transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
])
]

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
print('TTA: HorizontalFlip with crop')
print('validation with tta: ', evaluate_tta(model, val_loader, tta, 5000))
print('test with tta: ', evaluate_tta(model, test_loader, tta, 10000))
print('-'*50)
print()


tta = [
    transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
]), 
    transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(10),
]), 
    transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
]),
    transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
]),
    transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
    transforms.RandomRotation(10),
]),
    transforms.Compose([
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
]),
    transforms.Compose([
    transforms.RandomRotation(10),
])
]
train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
print('TTA: all combined')
print('validation with tta: ', evaluate_tta(model, val_loader, tta, 5000))
print('test with tta: ', evaluate_tta(model, test_loader, tta, 10000))
print('-'*50)
print()

"""# ResNet34"""

train_transform = transforms.Compose([                                      
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ResnetCNN(nn.Module):
    def __init__(self):

        super(ResnetCNN, self).__init__()
        self.pretrained_model = models.resnet34(pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, 10)
    
    def forward(self,x):
        return self.pretrained_model(x)

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)

model = ResnetCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_network(model, train_loader, val_loader, epochs = 15, print_results=True, 
            optimizer = optimizer, scheduler=scheduler)