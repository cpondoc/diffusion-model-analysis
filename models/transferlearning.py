# %pip install torchcam

'''
Finetuning existing ResNet model
'''
# Necessary Libraries
import torch.nn as nn
from torchvision import datasets, models, transforms

# Model Definition
def load_pretrained_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft
