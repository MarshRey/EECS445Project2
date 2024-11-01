import torch
import torch.nn as nn
import torch.nn.functional as F

class Challenge(nn.Module):
    def __init__(self):
        super(Challenge, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Adjust input size based on your image dimensions
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # Output layer for binary classification (Collie vs. Golden Retriever)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # Second convolutional block
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 256 * 8 * 8)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # Output layer without activation for CrossEntropyLoss

        return x