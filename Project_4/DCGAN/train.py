import argparse
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils

import models as models
from models.discriminator import DCGAN_Discriminator

from utils import train_networks, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="dcgan")
parser.add_argument("--data")
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--image-size", type=int, default=64)
parser.add_argument("--netD", default="", type=str)
parser.add_argument("--netG", default="", type=str)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--output_dir", default='runs', type=str)

def main():
    parameters = parser.parse_args()

    if parameters.seed is not None:
        random.seed(parameters.seed)
        torch.manual_seed(parameters.seed)
        cudnn.benchmark = True
        cudnn.deterministic = True

    # create model
    if parameters.pretrained:
        generator = models.__dict__[parameters.arch](pretrained=True)
    else:
        generator = models.__dict__[parameters.arch]()

    discriminator = DCGAN_Discriminator()
    
    dataloader = load_data(parameters.data, batch_size = parameters.batch_size, image_size = parameters.image_size)

    if parameters.netD != "":
        discriminator.load_state_dict(torch.load(parameters.netD))
    if parameters.netG != "":
        generator.load_state_dict(torch.load(parameters.netG))

    print('Start training')
    train_networks(dataloader, (generator, discriminator), 
        parameters.batch_size, parameters.epochs,
        parameters.output_dir)

if __name__ == "__main__":
    main()