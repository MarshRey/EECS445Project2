"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        ## TODO: DONE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.fc_1 = nn.Linear(in_features=32, out_features=8) # could be 8*2*2 of rinput size
        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            ## TODO: initialize the parameters for the convolutional layers
            ## TODO: DONE
            if conv is not None:
                # kaiming (He) initialization
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0) # Initialize bias to 0 but maybe could be 1
            ## TODO: initialize the parameters for [self.fc1] 
            ## TODO: DONE
            if self.fc_1 is not None:
                nn.init.kaiming_normal_(self.fc_1.weight, mode='fan_out', nonlinearity='relu')
                if self.fc_1.bias is not None:
                    nn.init.constant_(self.fc_1.bias, 0)
        
    def forward(self, x):
        N, C, H, W = x.shape

        # Hint: printing out x.shape after each layer could be helpful!
        ## TODO: forward pass
        ## TODO: DONE
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        print(f'After conv1: {x.shape}')

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        print(f'After conv2: {x.shape}')

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        print(f'After conv3: {x.shape}')

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc_1(x)
        print(f'After fc_1: {x.shape}')

        return x
        
