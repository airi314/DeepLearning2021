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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(train_transform, test_transform, random_generator):

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train, val = random_split(train, [45000, 5000], generator = random_generator)

    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def train_network(network, train_loader, val_loader, epochs = 10, criterion = None, optimizer = None):

    best_accuracy = -1
    best_model = None
    network.train_loss = list()
    network.train_accuracy = list()
    network.val_accuracy = list()
    network.epochs = epochs

    size_train = len(train_loader.dataset)
    size_val = len(val_loader.dataset)

    if not criterion:
      criterion = nn.CrossEntropyLoss()

    if not optimizer:
      optimizer = SGD(network.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):

        training_loss = 0
        training_correct = 0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs.detach(), dim=1)
            training_correct += (predicted == labels).sum().item()
        
        training_loss = training_loss/size_train
        training_accuracy = training_correct*100/size_train

        network.train_loss.append(training_loss)
        network.train_accuracy.append(training_accuracy)

        print(f"Epoch {epoch+1}")
        print(f"Training loss: {training_loss}")
        print(f"Training accuracy: {training_accuracy}%")
        
        validation_correct = 0
        with torch.no_grad():
            for inputs,labels in val_loader:
                outputs = network(inputs.to(device)).cpu()
                predicted = torch.argmax(outputs.detach(), dim=1)
                validation_correct += (predicted == labels).sum().item()
        
        val_accuracy = validation_correct*100/size_val
        network.val_accuracy.append(val_accuracy)

        print(f"Validation accuracy: {val_accuracy}%")

        if validation_correct > best_accuracy:
            best_accuracy = validation_correct
            best_model = deepcopy(network.state_dict())
          
        print('-'*30)

    print('Finished Training')

    network.best_model = best_model


def evaluate_network(network, test_loader):

    network.load_state_dict(network.best_model)
    size_test = len(test_loader.dataset)

    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = network(inputs.to(device)).cpu()
            predicted = torch.argmax(outputs, dim=1)
            test_correct += (predicted == labels).sum().item()

    print(f"Test accuracy: {test_correct*100/size_test}%")