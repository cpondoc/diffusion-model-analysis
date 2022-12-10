# ## CS 229 Final Project - Prompt Analysis
# Analyzing which prompts our network got wrong to better understand which prompts are adversarial.
#
# By: Christopher Pondoc, Joseph Guman, Joseph O'Brien

# ## Import Libraries
# Import all necessary libraries

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
from torch.autograd import Variable

# Joeys figuring stuff out
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image 
import PIL 

# ## Import Models
# Import all models (see `models/`).

# Import Models
from models.convbasic import ConvNeuralNet
from models.plainnet import PlainNet
from models.transferlearning import load_pretrained_model

# ## Dataset Class
# Handles Pre-Processing of all Images for CV Network

'''
Creating a Custom Image Dataset
'''
class ImageDataset(Dataset):

    def __init__(self, type_path=None, transform=None, percent=0.9):
        """
        Args:
            type_path: If either the train or test set
            transform (callable, optional): Optional transform to be applied
                on a sample.
            percent: how much percentage of the data to train on.
        """
        self.transform = transform
        self.type_path = type_path
        dalle_imgs = os.listdir('dataset/dalle')
        
        # Define the IDs (i.e., 00000, 01234) of the images in the chosen data.
        if (type_path is "train"):
            total_count = int((len(dalle_imgs) - 1) * percent)
            self.indices = [img[-9:-4] for img in dalle_imgs if (".jpg" in img)][:total_count]
        else:
            total_count = int((len(dalle_imgs) - 1) * 0.1)
            self.indices = [img[-9:-4] for img in dalle_imgs if (".jpg" in img)][-total_count:]
            
    def __len__(self):
        return len(self.indices) * 2

    def __getitem__(self, id):
        # Calculate whether real or DALLE
        data_half = int(id / len(self.indices))
        data_index = id % len(self.indices)
        
        # Find corresponding index of image
        img_id = self.indices[data_index]
        img_name = None
        if (data_half == 0):
            img_name = 'dataset/dalle/dalle-' + str(img_id) + '.jpg'
        else:
            img_name = 'dataset/real/real-' + str(img_id) + '.jpg'
        
        # Applying the image transformations and returning the data object
        torch_img = Image.open(img_name)
        if (self.transform):
            torch_img = self.transform(torch_img)
        return (torch_img, data_half, img_name)

# ## Helper Function to Plot Metrics
# Plot training and test accuracies.

def plot_metrics(metric_set, metric_name, save_path):
    fig, ax = plt.subplots()
    ax.plot(metric_set)
    ax.set(xlabel='epochs', ylabel=metric_name,
        title='Training ' + metric_name)
    fig.savefig('graphs/' + save_path)

# ## Helper Function to Test the Model
# Testing the model.

def generate_confusion_matrix(transform, weights_path, batch_size, network):
    # Loading in initial test data
    print("\nLoading in test data...")
    test_data = ImageDataset(type_path="test", transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in test data.")

    # Test Loading
    dataiter = iter(testloader)
    images, labels, img_names = next(dataiter)

    # Loading in a new example of the neural net, and loading in the weights
    net = network
    if (torch.cuda.is_available()):
        net.to('cuda')
    net.load_state_dict(torch.load(weights_path))

    # Getting accuracy of the data
    correct = 0
    total = 0
    
    # Generate heatmaps
    print("Generate Confusion Matrix")
    matrix = [[0, 0], [0, 0]]
    real_but_fake = []
    all_images = []
    with torch.no_grad():
        for data in testloader:
            images, labels, paths = data
            for i in range(len(paths)):
                all_images.append(paths[i])
            images_cuda, labels_cuda = images.cuda(), labels.cuda()
            outputs = net(images_cuda)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                prediction = int(predicted[i].item())
                actual = int(labels[i].item())
                matrix[prediction][actual] += 1
                if (prediction is 1 and actual is 0):
                    real_but_fake.append(paths[i])
    
    return matrix, real_but_fake, all_images

# ## Main Function
# Runs all of the necessary functions!

def main(model_type):
    # Map of all possible transforms
    data_transforms = {
        'ConvBasic': transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(250), # CHANGED FROM 250
         transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'TransferLearning': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(224),# CHANGED FROM 250
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # Map of all possible models
    models = {
        'ConvBasic': ConvNeuralNet(),
        'TransferLearning': load_pretrained_model()
    }
    model = models[model_type]
  
    # Batch size, max epochs, and threshold for convergence
    batch_size = 200
    max_epochs = 15
    threshold = 0.005
    
    # Look at different proportions of data and train + test accs
    proportions = [0.6]
    
    # Generate the necessary heatmaps
    matrix, real_but_fake, all_images = None, None, None
    for prop in proportions:
        PATH = 'weights/TransferLearning/dalle/TransferLearning-0.6.pth'
        matrix, real_but_fake, all_images = generate_confusion_matrix(data_transforms[model_type], PATH, batch_size, network=model)
    
    return matrix, real_but_fake, all_images

# ## Run all code!
# Runs all of the code for Transfer Learning.

matrix, real_but_fake, all_images = main(model_type = "TransferLearning")

print(matrix)

# ## Simplistic Linguistic Analysis
# Using NLTK to look at some simple data.

# +
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
from collections import Counter
import numpy as np
df = pd.read_csv('dataset/reference.csv')

total = 0
all_nouns = []
all_lengths = []
for img in all_images:
    index = int(img[-9:-4])
    description = df.iloc[index]['description']
    tokens = nltk.word_tokenize(description)
    tagged = nltk.pos_tag(tokens)
    counts = Counter(tag for word,tag in tagged)
    num_nouns = counts['NN'] + counts['NNS'] + counts['NNP'] + counts['NNPS']
    total += num_nouns
    all_nouns.append(num_nouns)
    all_lengths.append(len(tokens))

print("Entire Test Set:\n")
print("Number of Nouns:")
print("Mean: " + str(np.mean(all_nouns)))
print("Variance: " + str(np.var(all_nouns)))
print("")

print("Length of Message:")
print("Mean: " + str(np.mean(all_lengths)))
print("Variance: " + str(np.var(all_lengths)))
print("")

total = 0
rbf_nouns = []
rbf_lengths = []
print(len(real_but_fake))
for img in real_but_fake:
    print(img)
    index = int(img[-9:-4])
    description = df.iloc[index]['description']
    tokens = nltk.word_tokenize(description)
    tagged = nltk.pos_tag(tokens)
    counts = Counter(tag for word,tag in tagged)
    num_nouns = counts['NN'] + counts['NNS'] + counts['NNP'] + counts['NNPS']
    total += num_nouns
    rbf_nouns.append(num_nouns)
    rbf_lengths.append(len(tokens))
    
print("Images classified as real, but fake:\n")    
print("Number of Nouns:")
print("Mean: " + str(np.mean(rbf_nouns)))
print("Variance: " + str(np.var(rbf_nouns)))
print("")

print("Length of Message:")
print("Mean: " + str(np.mean(rbf_lengths)))
print("Variance: " + str(np.var(rbf_lengths)))
print("")
# +
#super awesome bootstrapping techniques
# -

# ## Looking for Specific Nouns
# We can then look for prompts that have these nouns to create an adversarial dataset.


noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
for img in real_but_fake:
    index = int(img[-9:-4])
    description = df.iloc[index]['description']
    tokens = nltk.word_tokenize(description)
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        if (tag in noun_tags):
            print(word)
    #counts = Counter(tag for word,tag in tagged)
    #num_nouns = counts['NN'] + counts['NNS'] + counts['NNP'] + counts['NNPS']
    #total += num_nouns
    #rbf_nouns.append(num_nouns)
    #rbf_lengths.append(len(tokens))

# ## Printing the Confusion Matrix

print(matrix)


