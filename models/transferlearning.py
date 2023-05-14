# %pip install torchcam

'''
Finetuning existing ResNet model
'''
# Necessary Libraries
import torch.nn as nn
from torchvision import datasets, models, transforms

# Model Definition
def load_pretrained_model():
    pretrained_model = models.resnet18(pretrained=False)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 2)
    return pretrained_model