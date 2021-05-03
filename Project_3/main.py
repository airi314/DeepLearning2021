from models import *
import torch
from torch.optim import SGD, Adam, lr_scheduler
from train import train_network, evaluate_network
from load_data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(train_loader, val_loader, test_loader), labels = load_data()

correct_labels = ['yes', 'no', 'up', 'down', 'left',
                  'right', 'on', 'off', 'stop', 'go', 'silence']

correct_index = [torch.tensor(labels.index(x)) for x in correct_labels]

model = M3(n_input=1, n_output=len(labels))
model.to(device)

optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

train_network(model, train_loader, val_loader,
              correct_index, 5, optimizer=optimizer)
