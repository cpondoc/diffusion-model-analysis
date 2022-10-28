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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

'''
Creating a Custom Image Dataset
'''
class ImageDataset(Dataset):

    def __init__(self, type_path=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.all_data = []

        # Load in all the DALLE images
        dalle_imgs = (os.listdir('data/' + type_path + '/dalle'))
        for image in dalle_imgs:
            img_name = 'data/' + type_path + '/dalle/' + image
            torch_img = io.imread(img_name)
            if (self.transform):
                torch_img = self.transform(torch_img)
            sample = {'image': torch_img, 'label': 1}
            self.all_data.append(sample)

        # Load in all the non-DALLE images
        non_dalle_imgs = (os.listdir('data/' + type_path + '/non-dalle'))
        for image in non_dalle_imgs:
            img_name = 'data/' + type_path + '/non-dalle/' + image
            torch_img = io.imread(img_name)
            if (self.transform):
                torch_img = self.transform(torch_img)
            sample = {'image': torch_img, 'label': 0}
            self.all_data.append(sample)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, id):
        return (self.all_data[id]["image"], self.all_data[id]["label"])

'''
Creating a Neural Network
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(250*250, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main():
    # Transform and batch size
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(250)])
    batch_size = 200

    # Loading in initial training data
    print("Loading in training data...")
    train_data = ImageDataset(type_path="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in training data.")

    # Loading in initial test data
    '''print("\nLoading in test data...")
    test_data = ImageDataset(type_path="test", transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in test data.")'''

    # Iterating through everything
    dataiter = iter(trainloader)
    features, labels = next(dataiter)
    print("Looking at one batch!")
    print(labels)
    print("\n")

    # Creating the CNN
    net = Net()

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Start Training!\n")
    for epoch in range(2):  # loop over the dataset multiple times
        print("Epoch " + str(epoch) + "\n")
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(running_loss)
            #if i % 500 == 499:    # print every 2000 mini-batches
            #    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
            #    running_loss = 0.0
    print('Finished Training')

    # Saving trained model
    PATH = 'weights/initial_training.pth'
    torch.save(net.state_dict(), PATH)

    
main()