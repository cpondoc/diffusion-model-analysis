# +
'''
Exporting the transforms for specific models
'''

# Necessary Libraries
import torchvision.transforms as transforms

# Dictionary mapping for specific transforms
data_transforms = {
    # Baseline CNN
    'ConvBasic': transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(250),
     transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    
    # Transfer Learning
    'TransferLearning': transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
