import pandas as pd
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # He normal initializer function
        def kaiming(layer):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Build layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # Dense layers
        self.fc1 = nn.Linear(128 * 80 * 80, 64)  
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # Apply He initialization
        self.apply(kaiming)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Conv block 3
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        return self.fc3(x)

