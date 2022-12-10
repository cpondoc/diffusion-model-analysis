# +
'''
Creating a Custom Image Dataset
'''

# Necessary Libraries
from torch.utils.data import Dataset
import os
from PIL import Image

class ImageDataset(Dataset):
    
    # Mapping between models, paths, and extensions
    model_map = {
        'dalle': {
            'extension': 'jpg',
            'prefix': 'dalle-'
        },
        'real': {
            'extension': 'jpg',
            'prefix': 'real-'
        },
        'stable-diffusion': {
            'extension': 'png',
            'prefix': 'stable-'
        },
    }
    
    '''
    Generates the IDs to define parity between the images.
    '''
    def __init__(self, type_path, transform, percent, first, second):
        # Define the transforms and whether it's test or train path
        self.transform = transform
        self.type_path = type_path
        
        # Define the two models we use the construct the dataset 
        self.first = first
        self.second = second
        model_imgs = os.listdir('dataset/' + self.first)
        
        # Define the IDs (i.e., 00000, 01234) of the images in the chosen data.
        if (type_path is "train"):
            total_count = int((len(model_imgs) - 1) * percent)
            self.indices = [img[-9:-4] for img in model_imgs if (self.model_map[self.first]['extension'] in img)][:total_count]
        else:
            total_count = int((len(model_imgs) - 1) * 0.1)
            self.indices = [img[-9:-4] for img in model_imgs if (self.model_map[self.first]['extension'] in img)][-total_count:]
    
    '''
    Returns the length of the number of images in the dataset.
    '''
    def __len__(self):
        return len(self.indices) * 2
    
    '''
    Returns the element at the specified index
    '''
    def __getitem__(self, id):
        # Calculate which class
        data_half = int(id / len(self.indices))
        data_index = id % len(self.indices)
        
        # Find corresponding index of image
        img_id = self.indices[data_index]
        img_name = None
        if (data_half == 0):
            img_name = 'dataset/' + self.first + '/' + self.model_map[self.first]['prefix'] + str(img_id) + '.' + self.model_map[self.first]['extension']
        else:
            img_name = 'dataset/' + self.second + '/' + self.model_map[self.second]['prefix'] + str(img_id) + '.' + self.model_map[self.second]['extension']
        
        # Applying the image transformations and returning the data object
        torch_img = Image.open(img_name)
        if (self.transform):
            torch_img = self.transform(torch_img)
        return (torch_img, data_half, img_name)
