# ## CS 229 Final Project - Training
# Notebook to set up all things related to training our ML models.
#
# By: Christopher Pondoc, Joseph Guman, Joseph O'Brien

# ## Libraries for Training
# Import all necessary libraries for training.

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

# ## Libraries for Heatmaps
# Import all necessary libraries for generating heatmaps in PyTorch.

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image 
import PIL 

# ## Import Models
# Import all models (see `models/`).

from models.convbasic import ConvNeuralNet
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
            img_name = 'dataset/stable-diffusion/stable-' + str(img_id) + '.png'
        
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

# ## Helper Function to Train Model
# Train the model.

def train_model(transform, batch_size, epochs, weights_path, model_type, network, threshold, proportion):
    # Loading in initial training data
    print("Proportion of Training Data: " + str(proportion * 100) + "\n")
    print("Loading in training data...")
    train_data = ImageDataset(type_path="train", transform=transform, percent=proportion)
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
    
    # Iterate for max epochs or until convergence
    curr_epoch = 0
    while (curr_epoch < 15 and ((curr_epoch < 8) or (abs(training_losses[-1] - training_losses[-2]) >= threshold))):
        # Reset the loss and correct
        print("Epoch " + str(curr_epoch + 1))
        running_loss = 0.0
        correct = 0

        # Iterate through each batch
        for i, data in enumerate(trainloader, 0):
            # Get batch data and zero parameter gradients
            inputs, labels, img_names = data
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
            running_loss += outputs.cpu().shape[0] * loss.item()
        
        # Calculation of the accuracy and loss
        total_loss = running_loss / len(train_data)
        accuracy = 100 * correct / len(train_data)
        print("Loss = {}".format(total_loss))
        print("Accuracy = {}\n".format(accuracy))

        # Appending to the arrays to look at further visualization
        training_losses.append(total_loss)
        training_accuracies.append(accuracy)
        curr_epoch += 1

    # Saving trained model
    print("Finished Training!\n")
    torch.save(net.state_dict(), weights_path)

    # Graph out training loss and accuracy over time
    plot_metrics(training_losses, 'Loss', model_type + '_' + str(proportion) + '_training_loss.png')
    plot_metrics(training_accuracies, 'Accuracy', model_type + '_' + str(proportion) + '_training_accuracies.png')
    return training_losses[-1], training_accuracies[-1]

# ## Helper Function to Test the Model
# Testing the model.

def test_model(transform, weights_path, batch_size, network):
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
    
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            images_cuda, labels_cuda = images.cuda(), labels.cuda()
            outputs = net(images_cuda)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print out accuracy!
    print('Accuracy of the network on the ' + str(len(test_data)) + ' test images: ' + str(100 * correct / total) + '%')
    return 100 * correct / total

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
    train_accs = []
    test_accs = []
    proportions = [0.6]
    
    # Train + test the data
    for prop in proportions:
        PATH = 'weights/' + model_type + '-' + str(prop) + '-dallevsd.pth'
        training_loss, training_accuracy = train_model(transform=data_transforms[model_type], batch_size=batch_size, epochs=max_epochs, weights_path=PATH, model_type=model_type, network=model, threshold=threshold, proportion=prop)
        test_accuracy = test_model(data_transforms[model_type], PATH, batch_size, network=model)
        
        # Update train and test accuracies
        train_accs.append(training_accuracy)
        test_accs.append(test_accuracy)
    return train_accs, test_accs

# ## Run all code!
# Runs all of the code for Transfer Learning.

# +
final_train = []
final_test = []

if __name__ == '__main__':
    train_accs, test_accs = main(model_type = "TransferLearning")
    
    final_train = train_accs
    final_test = test_accs
# -


