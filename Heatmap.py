# ## CS 229 Final Project - Heatmaps
# Code to set up heatmaps to gather insights from CNN training.
#
# By: Christopher Pondoc, Joseph Guman, Joseph O'Brien

# ## Import Libraries
# Import all necessary libraries

# First, import the necessary libraries
from __future__ import print_function, division
import gc
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
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
# In this case, we only need the pre-trained neural network.

from models.transferlearning import load_pretrained_model

# ## Import Dataset Class
# Used to help load in the images for heatmap generation

from utilities.dataset import ImageDataset


# ## Dataset Class
# Handles Pre-Processing of all Images for CV Network

# '''
# Creating a Custom Image Dataset
# '''
# class ImageDataset(Dataset):
#     
#     # Mapping between models, paths, and extensions
#     model_map = {
#         'dalle': {
#             'extension': 'jpg',
#             'prefix': 'dalle-'
#         },
#         'real': {
#             'extension': 'jpg',
#             'prefix': 'real-'
#         },
#         'stable-diffusion': {
#             'extension': 'png',
#             'prefix': 'stable-'
#         },
#     }
#     
#     '''
#     Generates the IDs to define parity between the images.
#     '''
#     def __init__(self, type_path, transform, percent, first, second):
#         # Define the transforms and whether it's test or train path
#         self.transform = transform
#         self.type_path = type_path
#         
#         # Define the two models we use the construct the dataset 
#         self.first = first
#         self.second = second
#         model_imgs = os.listdir('dataset/' + self.first)
#         
#         # Define the IDs (i.e., 00000, 01234) of the images in the chosen data.
#         if (type_path is "train"):
#             total_count = int((len(model_imgs) - 1) * percent)
#             self.indices = [img[-9:-4] for img in model_imgs if (".jpg" in img)][:total_count]
#         else:
#             total_count = int((len(model_imgs) - 1) * 0.1)
#             self.indices = [img[-9:-4] for img in model_imgs if (".jpg" in img)][-total_count:]
#     
#     '''
#     Returns the length of the number of images in the dataset.
#     '''
#     def __len__(self):
#         return len(self.indices) * 2
#     
#     '''
#     Returns the element at the specified index
#     '''
#     def __getitem__(self, id):
#         # Calculate which class
#         data_half = int(id / len(self.indices))
#         data_index = id % len(self.indices)
#         
#         # Find corresponding index of image
#         img_id = self.indices[data_index]
#         img_name = None
#         if (data_half == 0):
#             img_name = 'dataset/' + self.first + '/' + self.model_map[self.first]['prefix'] + str(img_id) + '.' + self.model_map[self.first]['extension']
#         else:
#             img_name = 'dataset/' + self.second + '/' + self.model_map[self.second]['prefix'] + str(img_id) + '.' + self.model_map[self.second]['extension']
#         
#         # Applying the image transformations and returning the data object
#         torch_img = Image.open(img_name)
#         if (self.transform):
#             torch_img = self.transform(torch_img)
#         return (torch_img, data_half, img_name)

# ## Function to Generate Heatmap
# Uses the same workflow as testing the network, but instead of inferencing on each image, we save a heatmap, instead.

def generate_heatmap(transform, weights_path, batch_size, network):
    # Loading in initial test data
    print("\nLoading in data...")
    test_data = ImageDataset(type_path="test", transform=transform, percent=0.6, first="dalle", second="stable-diffusion")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in data.")

    # Test Loading
    dataiter = iter(testloader)
    images, labels, img_names = next(dataiter)

    # Loading in a new example of the neural net, and loading in the weights
    net = network
    if (torch.cuda.is_available()):
        net.to('cuda')
    net.load_state_dict(torch.load(weights_path))
    
    # Iterate through all of the images and their names and generate heatmaps
    print("Generate Heatmaps...")
    cam_extractor = SmoothGradCAMpp(net)
    for data in testloader:
        images, labels, img_names = data
        images_cuda, labels_cuda = images.cuda(), labels.cuda()
        for image_cuda, img_name in zip(images_cuda, img_names):
                out = net(image_cuda.unsqueeze(0))
                activation_map = cam_extractor(out.cpu().squeeze(0).argmax().item(), out.cpu())

                current_image = Image.open(img_name)

                make_tensor = transforms.ToTensor()
                current_image = make_tensor(current_image)

                crop_tensor = transforms.CenterCrop(224)
                current_image = crop_tensor(current_image)

                result = overlay_mask(to_pil_image(current_image), to_pil_image(activation_map[0].cpu().squeeze(0), mode='F'), alpha=0.5)
                plt.imshow(result); plt.axis('off'); 
                plt.tight_layout();
                name = "heatmaps/test/" + "heatmap_of_"+img_name.split('/')[-1]
                plt.savefig(name)

# ## Main Function
# Runs all of the necessary functions!

def main(model_type):
    # Map of all possible transforms
    data_transforms = {
        'TransferLearning': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(224),# CHANGED FROM 250
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # Map of all possible models
    models = {
        'TransferLearning': load_pretrained_model()
    }
    model = models[model_type]
  
    # Batch size
    batch_size = 200
    
    # Generate the necessary heatmaps
    PATH = 'weights/TransferLearning-0.6-dallevsd.pth'
    generate_heatmap(data_transforms[model_type], PATH, batch_size, network=model)

# ## Run all code!
# Runs all of the code for Transfer Learning.

if __name__ == '__main__':
    main(model_type = "TransferLearning")




