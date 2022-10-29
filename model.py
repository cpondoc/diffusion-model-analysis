'''
Initial training of the data
'''

# First, import the necessary libraries
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import Dataset
from torchvision import utils

from PIL import Image

from models.convbasic import ConvNeuralNet
from models.plainnet import PlainNet

'''
Creating a Custom Image Dataset
'''
class ImageDataset(Dataset):

    def __init__(self, type_path=None, transform=None):
        """
        Args:
            type_path: If either the train or test set
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.all_data = []

        # Load in all the DALLE images
        dalle_imgs = (os.listdir('data/' + type_path + '/dalle'))
        for image in dalle_imgs:
            img_name = 'data/' + type_path + '/dalle/' + image
            torch_img = Image.open(img_name)
            #torch_img = io.imread(img_name)
            if (self.transform):
                torch_img = self.transform(torch_img)
            sample = {'image': torch_img, 'label': 1}
            self.all_data.append(sample)

        # Load in all the non-DALLE images
        non_dalle_imgs = (os.listdir('data/' + type_path + '/non-dalle'))
        for image in non_dalle_imgs:
            img_name = 'data/' + type_path + '/non-dalle/' + image
            torch_img = Image.open(img_name)
            #torch_img = io.imread(img_name)
            if (self.transform):
                torch_img = self.transform(torch_img)
            sample = {'image': torch_img, 'label': 0}
            self.all_data.append(sample)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, id):
        return (self.all_data[id]["image"], self.all_data[id]["label"])

'''
Create plots for training and test accuracies for several epochs.
'''
def plot_metrics(metric_set, metric_name, save_path):
    fig, ax = plt.subplots()
    ax.plot(metric_set)
    ax.set(xlabel='epochs', ylabel=metric_name,
        title='Training ' + metric_name)
    fig.savefig('graphs/' + save_path)

'''
Training the model!
'''
def train_model(transform, batch_size, epochs, weights_path):
    # Loading in initial training data
    print("Loading in training data...")
    train_data = ImageDataset(type_path="test", transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in training data.\n")

    # Creating the CNN, loss function, and optimizer
    net = ConvNeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train for number of epochs
    print("Start Training!\n")
    training_losses = []
    training_accuracies = []
    for epoch in range(epochs):

        # Reset the loss and correct
        print("Epoch " + str(epoch + 1))
        running_loss = 0.0
        correct = 0

        # Iterate through each batch
        for i, data in enumerate(trainloader, 0):
            # Get batch data and zero parameter gradients
            inputs, labels = data
            optimizer.zero_grad()

            # Forward + Backward Propopagation
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Optimization Step
            optimizer.step()

            # Calculate accuracy and loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum()
            running_loss += outputs.shape[0] * loss.item()
        
        # Calculation of the accuracy and loss
        total_loss = running_loss / len(train_data)
        accuracy = 100 * correct / len(train_data)
        print("Loss = {}".format(total_loss))
        print("Accuracy = {}\n".format(accuracy))

        # Appending to the arrays to look at further visualization
        training_losses.append(total_loss)
        training_accuracies.append(accuracy)

    # Saving trained model
    print("Finished Training!\n")
    torch.save(net.state_dict(), weights_path)

    # Graph out training loss and accuracy over time
    plot_metrics(training_losses, 'Loss', 'cnn_training_loss.png')
    plot_metrics(training_accuracies, 'Accuracy', 'cnn_training_accuracies.png')

'''
Testing the model!
'''
def test_model(transform, weights_path, batch_size):
    # Loading in initial test data
    print("\nLoading in test data...")
    test_data = ImageDataset(type_path="test", transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in test data.")

    # Test Loading
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Loading in a new example of the neural net, and loading in the weights
    net = ConvNeuralNet()
    net.load_state_dict(torch.load(weights_path))

    # Getting accuracy of the data
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print out accuracy!
    print('Accuracy of the network on the ' + str(len(test_data)) + ' test images: ' + str(100 * correct // total) + '%')

'''
Main hub of setting the hyperparameters, and then calling training and testing for the model.
'''
def main():
    # Transform and batch size
    '''transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(250)])'''
    transform = transforms.Compose(
    [ transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(250),
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 200

    # Training and testing the model
    PATH = 'weights/cnn_training.pth'
    train_model(transform=transform, batch_size=batch_size, epochs=5, weights_path=PATH)
    test_model(transform, PATH, batch_size)

if __name__ == '__main__':
    main()