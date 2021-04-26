# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.optim import SGD, Adam, lr_scheduler
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dataset import SPEECHCOMMANDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# load and split data
train_set = SPEECHCOMMANDS(root = 'data/train/', subset='training')
valid_set = SPEECHCOMMANDS(root = 'data/train/', subset='validation')
test_set = SPEECHCOMMANDS(root = 'data/train/', subset='testing')

# check dataset size
print('Number of train instances: ', len(train_set))
print('Number of validation instances: ', len(valid_set))
print('Number of test instances: ', len(test_set))

labels_train = sorted(list(set(x[2] for x in train_set)))
print('Labels: ', labels_train)

labels_test = sorted(list(set(x[2] for x in test_set)))
print('Labels: ', labels_test)

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

batch_size = 64

# map label name to index in the array
def label_to_index(word):
    return torch.tensor(labels_train.index(word))

# pad data with zeros - each item should be same length
def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

# prepare data
def collate_fn(batch):

    inputs, labels = [], []

    for waveform, _, label, *_ in batch:
        inputs += [waveform]
        labels += [label_to_index(label)]

    inputs = pad_sequence(inputs)
    labels = torch.stack(labels)

    return inputs, labels


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

val_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)