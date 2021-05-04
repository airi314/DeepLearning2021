# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.optim import SGD, Adam, lr_scheduler
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dataset import SPEECHCOMMANDS
import numpy as np
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def load_data(root='data/train', batch_size=64):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic=True

    # load and split data
    train_set = SPEECHCOMMANDS(root, subset='training')
    valid_set = SPEECHCOMMANDS(root, subset='validation')
    test_set = SPEECHCOMMANDS(root, subset='testing')

    # check dataset size
    print('Number of train instances: ', len(train_set))
    print('Number of validation instances: ', len(valid_set))
    print('Number of test instances: ', len(test_set))

    labels = sorted(list(set(x[2] for x in train_set)))
    labels = ['unknown'] + labels

    if DEVICE == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, labels),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )

    val_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, labels),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, labels),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )

    return (train_loader, val_loader, test_loader), labels


# pad data with zeros - each item should be same length
def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

# prepare data
def collate_fn(batch, labels_train):
    inputs, labels = [], []
    for waveform, _, label, *_ in batch:
        inputs += [waveform]
        labels += [torch.tensor(labels_train.index(label))]
    inputs = pad_sequence(inputs)
    labels = torch.stack(labels)
    return inputs, labels
