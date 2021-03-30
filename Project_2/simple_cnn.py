# %%
"""
# Importing libraries
"""

# %%
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

# %%
train_loader, val_loader, test_loader = load_data(train_transform, test_transform, random_generator)

# %%
"""
# Exploring dataset
"""

# %%
data_iterator = iter(train_loader)
images, classes = data_iterator.next()

# %%
images[0].shape

# %%
fig = plt.figure(figsize=(25, 8))

def show_image(image):
    image = image / 2 + 0.5
    image = image.numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))

n_row = 3
batch_size = 8

for row_idx in np.arange(n_row):
    images, classes = data_iterator.next()
    for idx in np.arange(batch_size):
      ax = fig.add_subplot(n_row, batch_size, batch_size*row_idx+idx+1, xticks=[], yticks=[])
      show_image(images[idx])
      ax.set_title(classes_names[classes[idx]])

# %%
"""
# Simple CNN model with SGD optimizer
"""

# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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

# %%
model = SimpleCNN().to(device)
# optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# %%
train_network(model, train_loader, val_loader, epochs = 30)

# %%
evaluate_network(model, test_loader)

# %%
plt.figure()
plt.plot([i for i in range(model.epochs)], model.train_accuracy)
plt.plot([i for i in range(model.epochs)], model.val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.title('Accuracy for train and test set for simple CNN model with SGD optimizer')
plt.show()

# %%
plt.figure()
plt.plot([i for i in range(model.epochs)], model.train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss for train and for simple CNN model with SGD optimizer')
plt.show()

# %%
"""
# Simple CNN model with Adam optimizer
"""

# %%
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# %%
train_network(model, train_loader, val_loader, optimizer = optimizer, epochs = 30)

# %%
evaluate_network(model, test_loader)

# %%
plt.figure()
plt.plot([i for i in range(model.epochs)], model.train_accuracy)
plt.plot([i for i in range(model.epochs)], model.val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.title('Accuracy for train and test set for simple CNN model with Adam optimizer')
plt.show()

# %%
plt.figure()
plt.plot([i for i in range(model.epochs)], model.train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss for train and for simple CNN model with Adam optimizer')
plt.show()

# %%
"""
# Simple CNN model with Adam optimizer and simple data augmentation
"""

# %%
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=.40),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
train_network(model, train_loader, val_loader, optimizer = optimizer, epochs = 30)

# %%
evaluate_network(model, test_loader)

# %%
plt.figure()
plt.plot([i for i in range(model.epochs)], model.train_accuracy)
plt.plot([i for i in range(model.epochs)], model.val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.title('Accuracy for train and test set for simple CNN model with Adam optimizer')
plt.show()

# %%
plt.figure()
plt.plot([i for i in range(model.epochs)], model.train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss for train and for simple CNN model with Adam optimizer')
plt.show()

# %%
