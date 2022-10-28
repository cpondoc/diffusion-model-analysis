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

class DalleDataset(Dataset):
    """DALLE dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.all_data = []
        dalle_folders = (os.listdir('data/dalle'))
        for folder in dalle_folders:
            if (folder[0] != "."):
                images = (os.listdir('data/dalle/' + folder))
                for image in images:
                    img_name = 'data/dalle/' + folder + '/' + image
                    torch_img = io.imread(img_name)
                    sample = {'image': torch_img, 'label': 1}
                    self.all_data.append(sample)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, id):
        if (self.transform):
            img = self.transform(self.all_data[id]["image"])
            return (img, self.all_data[id]["label"])
        return (self.all_data[id]["image"], self.all_data[id]["label"])

def main():
    # Data transformation -- turn into a PyTorch Tensor, normalize the data (mean, std)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Loading in initial training data
    dalle_data = DalleDataset(transform=transform)
    batch_size = 20
    trainloader = torch.utils.data.DataLoader(dalle_data, batch_size=batch_size,
                                            shuffle=True)

    # Iterating through everything
    dataiter = iter(trainloader)
    features, labels = next(dataiter)
    print(labels)

main()