'''
Creating a Neural Network
'''
# Necessary Libraries
import torch.nn as nn

# Class Definition
class PlainNet(nn.Module):
    def __init__(self):
        super(PlainNet, self).__init__()
        self.flatten = nn.Flatten()
        '''self.linear_relu_stack = nn.Sequential(
            nn.Linear(250*250, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )'''

    def forward(self, x):
        x = self.flatten(x)
        return x
        print(x.shape)
        #logits = self.linear_relu_stack(x)
        #return logits


