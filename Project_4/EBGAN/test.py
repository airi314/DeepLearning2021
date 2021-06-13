import argparse
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as utils

import models

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="dcgan")
parser.add_argument("--num-images", type=int, default=64)
parser.add_argument("--model-path", default=None, type=str)
parser.add_argument("--pretrained", dest="pretrained", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--save-single", action="store_true")
parser.add_argument('--output-dir', default = 'tests')

def main():
    parameters = parser.parse_args()

    random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)

    if parameters.pretrained:
        model = models.__dict__[parameters.arch](pretrained=True)
    else:
        model = models.__dict__[parameters.arch]()

    if parameters.model_path is not None:
        model.load_state_dict(torch.load(parameters.model_path, map_location=torch.device("cpu")))
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(0)    

    model = model.to(device)
    cudnn.benchmark = True
    cudnn.deterministic = True

    noise = torch.randn([parameters.num_images, 100, 1, 1])
    noise = noise.to(device, non_blocking=True)

    with torch.no_grad():
        generated_images = model(noise)
    if not os.path.exists(parameters.output_dir):
        os.makedirs(parameters.output_dir)
    if parameters.save_single:
        for i, image in enumerate(generated_images):
            utils.save_image(image, os.path.join(parameters.output_dir, str(i)+".png"), normalize=True)
    else:
        utils.save_image(generated_images, os.path.join(parameters.output_dir, "test.png"), normalize=True)

if __name__ == "__main__":
    main()