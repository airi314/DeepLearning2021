import os
from models import *
import torch
from torch.optim import Adam, RMSprop, lr_scheduler
from train import train_network, evaluate_network, get_predictions, plot_confusion_matrix
from load_data import load_data
from utils import *
                                 
train_accuracies = list()
val_accuracies = list()
correct_labels = ['yes', 'no', 'up', 'down', 'left',
                'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
correct_index = [torch.tensor(labels.index(x)) for x in correct_labels]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for optim in [Adam, RMSprop]:
	accuracies = list()
	for lr in [0.01, 0.001, 0.0001]:

		(train_loader, val_loader, test_loader), labels = load_data(root='data/train')

		model = architecture(n_input=1, n_output=len(labels))
		model.to(device)

		optimizer = optim(model.parameters(), lr=lr, weight_decay=0.0001)

		train_network(model, train_loader, val_loader,
		            correct_index, 10, optimizer=optimizer, print_results=False)
		
		accuracies.append(model.train_accuracy)
		accuracies.append(model.val_accuracy)

		print('Optimizer: ', str(optim), ', learning rate: ', str(lr))
		print('Accuracy on the validation set: ', evaluate_network(model, val_loader, correct_index))
		print('Accuracy on the test set: ', evaluate_network(model, test_loader, correct_index))
		print('-' * 20)

	plt.figure()
	for acc, color in zip(accuracies, ['r-', 'r--', 'g-', 'g--', 'b-', 'b--']:
		plt.plot([i for i in range(epoch)], acc, color)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend(['Train, 0.01', 'Valid, 0.01', 'Train, 0.001', 'Valid, 0.001','Train, 0.0001', 'Valid, 0.0001',])
	plt.title('Accuracy during training for different learning rates')
	plt.savefig(str(optim) + '_cnn.png')