import matplotlib.pyplot as plt

def plot_loss(network, epoch = 10):
    plt.figure()
    plt.plot([i for i in range(epoch)], network.train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss for train set')
    plt.show()


def plot_accuracy(network, epoch = 10):
    plt.figure()
    plt.plot([i for i in range(epoch)], network.train_accuracy)
    plt.plot([i for i in range(epoch)], network.val_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Accuracy for train and test set')
    plt.show()

