import torch
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


def map_predictions(correct_index, predicted):
    return [x.item() if x in correct_index else 0 for x in predicted]


def train_network(network,
                  train_loader, val_loader, correct_index, epochs=10,
                  criterion=None, optimizer=None,
                  transforms_train=list(), transforms_val=list(),
                  print_results = False):

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
        if print_results:
            print('Loss on training set: ', training_loss)
        training_accuracy = training_correct*100/size_train
        if print_results:
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
        if print_results:
            print('Accuracy on validation set: ', val_accuracy)
            print('-'*20)

        network.val_accuracy.append(val_accuracy)

        best_model = deepcopy(network.state_dict())
        network.best_model = best_model


def evaluate_network(network, test_loader, correct_index):

    network.load_state_dict(network.best_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_correct = 0
    size_test = len(test_loader.dataset)

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        output = network(data)

    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = network(inputs.to(device)).cpu()
            predicted = torch.argmax(outputs.detach(), dim=2)
            predicted = map_predictions(correct_index, predicted)
            test_correct += (np.array(predicted) == np.array(labels)).sum()

    print('Accuracy o test set: ', test_correct*100/size_test)


def get_predictions(network, test_loader, correct_index):

    y_true = np.array([])
    y_pred = np.array([])

    network.load_state_dict(network.best_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_correct = 0
    size_test = len(test_loader.dataset)

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        output = network(data)

    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = network(inputs.to(device)).cpu()
            predicted = torch.argmax(outputs.detach(), dim=2)
            predicted = map_predictions(correct_index, predicted)
            test_correct += (np.array(predicted) == np.array(labels)).sum()
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, predicted))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, correct_index, correct_labels = None):

    cm = confusion_matrix(y_true, y_pred, correct_index)
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')

    if correct_labels is not None:
        tick_marks = np.arange(len(correct_labels))
        plt.xticks(tick_marks, correct_labels, rotation=45)
        plt.yticks(tick_marks, correct_labels)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()