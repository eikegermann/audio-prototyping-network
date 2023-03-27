import os
import copy
import math
import random
import librosa
import librosa.display
import itertools
import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython.display as ipd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from IPython.display import Audio, display
from torchsummary import summary

class AudioClassifier(nn.Module):
    def __init__(self, n_bands=96, n_frames=97, n_features=3):
        super(AudioClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1a = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.4)

        self.fc1 = None
        self.fc2 = nn.Linear(5120, 1280)
        self.fc2a = nn.Linear(1280, 768)
        self.fc2b = nn.Linear(768, 512)
        self.fc2c = nn.Linear(512, 256)
        self.fc2d = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_features)
        
        self._initialize_fc1(n_bands, n_frames)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn1a(self.conv1a(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2a(x))
        x = self.dropout(x)
        x = F.relu(self.fc2b(x))
        x = F.relu(self.fc2c(x))
        x = F.relu(self.fc2d(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    
    def _initialize_fc1(self, n_bands, n_frames):

        with torch.no_grad():
            sample_input = torch.randn(1, 1, n_bands, n_frames)
            # print("Training Sample Input Shape:", sample_input.shape)
            x = self.pool(F.relu(self.bn1(self.conv1(sample_input))))
            x = self.pool(F.relu(self.bn1a(self.conv1a(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            # print("Training X Shape:", x.shape)
            flattened_size = x.view(x.size(0), -1).shape[1]
            # print(flattened_size)
            self.fc1 = nn.Linear(flattened_size, 5120)