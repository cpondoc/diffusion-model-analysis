'''
Initial training of the data
'''

# First, import the necessary libraries
from __future__ import print_function, division
import gc
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import utils
import torchvision.transforms as transforms

# Import Models
from models.convbasic import ConvNeuralNet
from models.plainnet import PlainNet
from models.transferlearning import load_pretrained_model

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
        self.type_path = type_path
        self.dalle_length = len(os.listdir('data/' + type_path + '/dalle'))

    def __len__(self):
        return self.dalle_length * 2

    def __getitem__(self, id):
        if (id < self.dalle_length):
            dalle_imgs = (os.listdir('data/' + self.type_path + '/dalle'))
            image = dalle_imgs[id]
            img_name = 'data/' + self.type_path + '/dalle/' + image
            torch_img = Image.open(img_name)
            if (self.transform):
                torch_img = self.transform(torch_img)
            sample = {'image': torch_img, 'label': 1}
            return (sample['image'], sample['label'])
        else:
            non_dalle_imgs = (os.listdir('data/' + self.type_path + '/non-dalle'))
            image = non_dalle_imgs[id - self.dalle_length]
            img_name = 'data/' + self.type_path + '/non-dalle/' + image
            torch_img = Image.open(img_name)
            if (self.transform):
                torch_img = self.transform(torch_img)
            sample = {'image': torch_img, 'label': 0}
            return (sample['image'], sample['label'])

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
def train_model(transform, batch_size, epochs, weights_path, model_type, network):
    # Loading in initial training data
    print("Loading in training data...")
    train_data = ImageDataset(type_path="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in training data.\n")

    # Creating the CNN, loss function, and optimizer
    net = network
    if (torch.cuda.is_available()):
        net.to('cuda')
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
            inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # Forward + Backward Propopagation
            outputs = net(inputs_cuda)
            loss = criterion(outputs, labels_cuda)
            loss.backward()

            # Optimization Step
            optimizer.step()

            # Calculate accuracy and loss
            _, predicted = torch.max(outputs.cpu().data, 1)
            correct += (predicted == labels).float().sum()
            #_, predicted = torch.max(outputs.cpu().data, 1)
            running_loss += outputs.cpu().shape[0] * loss.item()
        
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
    plot_metrics(training_losses, 'Loss', model_type + '_training_loss.png')
    plot_metrics(training_accuracies, 'Accuracy', model_type + '_training_accuracies.png')

'''
Testing the model!
'''
def test_model(transform, weights_path, batch_size, network):
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
    net = network
    if (torch.cuda.is_available()):
        net.to('cuda')
    net.load_state_dict(torch.load(weights_path))

    # Getting accuracy of the data
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images_cuda, labels_cuda = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(images_cuda)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print out accuracy!
    print('Accuracy of the network on the ' + str(len(test_data)) + ' test images: ' + str(100 * correct // total) + '%')

'''
Main hub of setting the hyperparameters, and then calling training and testing for the model.
'''
def main(model_type):
    # Map of all possible transforms
    data_transforms = {
        'PlainNet': transforms.Compose(
        [transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(250)]),
        'ConvBasic': transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(250),
         transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'TransferLearning': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(250),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # Map of all possible models
    models = {
        'PlainNet': PlainNet(),
        'ConvBasic': ConvNeuralNet(),
        'TransferLearning': load_pretrained_model()
    }
    model = models[model_type]
  
    # Batch size
    batch_size = 200

    # Training and testing the model
    PATH = 'weights/' + model_type + '.pth'
    train_model(transform=data_transforms[model_type], batch_size=batch_size, epochs=5, weights_path=PATH, model_type=model_type, network=model)
    test_model(data_transforms[model_type], PATH, batch_size, network=model)

'''
Run main and then perform everything
'''
if __name__ == '__main__':
    models = ["ConvBasic", "TransferLearning"]
    for model in models:
        main(model_type=model)
