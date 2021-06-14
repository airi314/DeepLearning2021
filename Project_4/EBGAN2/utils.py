import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as torch_utils

from tqdm import tqdm
import os
import numpy as np


def load_data(data_path, data_transforms = None, batch_size = 64, image_size = 64):

    if data_transforms is None:
        data_transforms = transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])

    dataset = ImageFolder(root = data_path, transform= data_transforms)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=2,
                                             drop_last=True)
    return dataloader

def pullaway_loss(embeddings):
    norm = torch.norm(embeddings, 1)
    normalized_embeddings = embeddings / norm
    similarity = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(1, 0)) ** 2
    batch_size = embeddings.size()[0]
    pt_loss = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return pt_loss

def train_networks(dataloader, networks, batch_size, epochs, output_dir, pt_loss_weight=0.1, margin=1):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator, discriminator = networks
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    criterion = nn.MSELoss().to(device)
    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) # values recommended for dcgan
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    base_image = torch.randn(batch_size, 100, 1, 1).to(device)

    d_loss_list = list()
    g_loss_list = list()

    discriminator.train()

    for epoch in tqdm(range(1, epochs+1)):

        generator.train()

        for i, (inputs, _) in enumerate(dataloader):

            inputs = inputs.to(device, non_blocking=True)

            noise = torch.randn(batch_size, 100, 1, 1).to(device, non_blocking=True) 

            # update discriminator
            optim_D.zero_grad()
            real_output,_ = discriminator(inputs)
            d_loss_real = criterion(real_output, inputs)
            d_loss_real.backward()

            fake_imgs = generator(noise)
            fake_output,_ = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_imgs.detach())
            d_loss_fake.backward()

            d_loss = d_loss_real + torch.clamp(margin - d_loss_fake, min=0)
            optim_D.step()

            # update generator
            optim_G.zero_grad()
            fake_output, embeddings = discriminator(fake_imgs)
            g_loss_fake = criterion(fake_output, fake_imgs.detach())
            g_loss = g_loss_fake + pt_loss_weight * pullaway_loss(embeddings)

            g_loss.backward()
            optim_G.step()

            iters = i + (epoch-1) * len(dataloader) + 1

            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

        print("d_loss:", d_loss.item())
        print("g_loss:", g_loss.item(), iters)

        for path in [output_dir, os.path.join(output_dir, 'images2'), os.path.join(output_dir, 'weights2')]:
            if not os.path.exists(path):
                os.makedirs(path)

        with torch.no_grad():
            generated_batch = generator(base_image)
            torch_utils.save_image(generated_batch.detach(), os.path.join(os.path.join(output_dir, 'images2'), f"GAN_epoch_{epoch}.png"), normalize=True)

        torch.save(generator.state_dict(), os.path.join(os.path.join(output_dir, 'weights2'), f"Generator_epoch{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(os.path.join(output_dir, 'weights2'), f"Discriminator_epoch{epoch}.pth"))

        d_loss_array = np.array(d_loss_list)
        g_loss_array = np.array(g_loss_list)

        np.save(os.path.join(output_dir, "d_loss_2.npy"), d_loss_array)
        np.save(os.path.join(output_dir, "g_loss_2.npy"), g_loss_array)