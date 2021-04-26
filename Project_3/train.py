import torch
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
import numpy as np


def map_predictions(correct_index, predicted):
    return [x.item() if x in correct_index else 0 for x in predicted]


def train_network(network,
                  train_loader, val_loader, correct_index, epochs=10,
                  criterion=None, optimizer=None,
                  transforms_train=list(), transforms_val=list()
                  ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network.train_loss = list()
    network.train_accuracy = list()
    network.val_accuracy = list()

    size_train = len(train_loader.dataset)
    size_val = len(val_loader.dataset)

    if criterion is None:
        criterion = F.nll_loss

    if optimizer is None:
        optimizer = SGD(network.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(epochs)):

        network.train(True)

        training_loss = 0
        training_correct = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            for t in transforms_train:
                inputs = t(inputs)

            optimizer.zero_grad()
            outputs = network(inputs)

            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs.detach(), dim=2)
            training_correct += (predicted ==
                                 labels.reshape(-1, 1)).sum().item()

        training_loss = training_loss/size_train
        print('Loss on validation set: ', training_loss)
        training_accuracy = training_correct*100/size_train
        print('Accuracy on training set: ', training_accuracy)

        network.train_loss.append(training_loss)
        network.train_accuracy.append(training_accuracy)

        validation_correct = 0
        network.train(False)
        with torch.no_grad():
            for inputs, labels in val_loader:
                for t in transforms_val:
                    inputs = t(inputs)
                outputs = network(inputs.to(device)).cpu()
                predicted = torch.argmax(outputs.detach(), dim=2)
                predicted = map_predictions(correct_index, predicted)
                validation_correct += (np.array(predicted)
                                       == np.array(labels)).sum()

        val_accuracy = validation_correct*100/size_val
        print('Accuracy on validation set: ', val_accuracy)
        print('-'*20)
        network.val_accuracy.append(val_accuracy)


def evaluate_network(network, test_loader, correct_index):

    test_correct = 0
    size_test = len(test_loader.dataset)

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)

    test_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = network(inputs.to(device)).cpu()
            predicted = torch.argmax(outputs.detach(), dim=2)
            predicted = map_predictions(correct_index, predicted)
            test_correct += (np.array(predicted) == np.array(labels)).sum()

    return test_correct*100/size_test
