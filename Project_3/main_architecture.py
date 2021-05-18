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

for architecture in [M3, M5, M11, M18]:

	(train_loader, val_loader, test_loader), labels = load_data(root='data/train')

	model = architecture(n_input=1, n_output=len(labels))
	model.to(device)

	optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

	train_network(model, train_loader, val_loader,
	            correct_index, 10, optimizer=optimizer, print_results=False)
	train_accuracies.append(model.train_accuracy)
	val_accuracies.append(model.val_accuracy)

	print('Architecture: ', str(architecture))
	print('Accuracy on the validation set: ', evaluate_network(model, val_loader, correct_index))
	print('Accuracy on the test set: ', evaluate_network(model, test_loader, correct_index))
	print('-' * 20)

plt.figure()
for accuracies in train_accuracies:
	plt.plot([i for i in range(epoch)], accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['M3', 'M5', 'M11', 'M18'])
plt.title('Accuracy during training for different network architectures')
plt.savefig('architecture_train.png')

plt.figure()
for accuracies in val_accuracies:
	plt.plot([i for i in range(epoch)], accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['M3', 'M5', 'M11', 'M18'])
plt.title('Accuracy during training for different network architectures')
plt.savefig('architecture_val.png')