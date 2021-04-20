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
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_generator = torch.manual_seed(0)


def evaluate(model, test_loader, size):
    model.load_state_dict(model.best_model)
    data_iterator = iter(test_loader)
    test_correct = 0
    for i in range((size//64)+1):
        images, classes = data_iterator.next()
        with torch.no_grad():
            outputs = model(images.to(device)).cpu()
            predicted = torch.argmax(outputs, dim=1)
            test_correct += (predicted == classes).sum().item()
    return test_correct/size

def evaluate_tta(model, test_loader, tta_transforms, size):
    model.load_state_dict(model.best_model)
    data_iterator = iter(test_loader)
    test_correct = 0
    for i in range((size//64)+1):
        images, classes = data_iterator.next()
        outputs = model(images.to(device)).cpu()
        for transforms in tta_transforms:
            transformed_images = transforms(images)
            with torch.no_grad():
                outputs += model(transformed_images.to(device)).cpu()
        predicted = torch.argmax(outputs, dim=1)
        test_correct += (predicted == classes).sum().item()
    return test_correct/size