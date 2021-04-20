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

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_generator = torch.manual_seed(0)

# %%
classes_names = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 9, 3)
        self.fc1 = nn.Linear(9 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 9 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
model = CNN().to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
print('optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)')
train_network(model, train_loader, val_loader, epochs = 50, print_results=False, optimizer=optimizer)
print(evaluate_network(model, val_loader))
print(evaluate_network(model, test_loader))
print('-'*50)
print()

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
model = CNN().to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
print('optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)')
train_network(model, train_loader, val_loader, epochs = 50, print_results=False, optimizer=optimizer)
print(evaluate_network(model, val_loader))
print(evaluate_network(model, test_loader))
print('-'*50)
print()

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
model = CNN().to(device)
optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
print('optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)')
train_network(model, train_loader, val_loader, epochs = 50, print_results=False, optimizer=optimizer)
print(evaluate_network(model, val_loader))
print(evaluate_network(model, test_loader))
print('-'*50)
print()


train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
model = CNN().to(device)
optimizer = Adam(model.parameters(), lr=0.01)
print('optimizer = Adam(model.parameters(), lr=0.01)')
train_network(model, train_loader, val_loader, epochs = 50, print_results=False, optimizer=optimizer)
print(evaluate_network(model, val_loader))
print(evaluate_network(model, test_loader))
print('-'*50)
print()

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
model = CNN().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
print('optimizer = Adam(model.parameters(), lr=0.001)')
train_network(model, train_loader, val_loader, epochs = 50, print_results=False, optimizer=optimizer)
print(evaluate_network(model, val_loader))
print(evaluate_network(model, test_loader))
print('-'*50)
print()

train_loader, val_loader, test_loader = load_data(train_transform, test_transform)
model = CNN().to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
print('optimizer = Adam(model.parameters(), lr=0.0001)')
train_network(model, train_loader, val_loader, epochs = 50, print_results=False, optimizer=optimizer)
print(evaluate_network(model, val_loader))
print(evaluate_network(model, test_loader))
print('-'*50)
print()
