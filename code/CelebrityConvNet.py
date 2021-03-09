# Python Module CelebrityConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CelebrityConvNet(nn.Module):
    def __init__(self, image_size, num_classes):
        super(CelebrityConvNet, self).__init__() 
        # Input image_sizeximage_sizex3 image
        # 16 filters
        # 7x7 filter size 
        # stride 2 (downsampling by factor of 2)
        # Output image: image_size/2ximage_size/2x16
        self.conv1 = nn.Conv2d(3, 16, 7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Input image_size/2ximage_size/2x16 image
        # 32 filters
        # 3x3 filter size 
        # stride 2 (downsampling by factor of 2)
        # Output image: image_size/4ximage_size/4x32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Input image_size/4ximage_size/4x32 image
        # 64 filters
        # 3x3 filter size 
        # stride 2 (downsampling by factor of 2)
        # Output image: image_size/8ximage_size/8x64
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)     
        self.bn3 = nn.BatchNorm2d(64)

        # Input image_size/8ximage_size/8x64 image
        # 128 filters
        # 3x3 filter size 
        # Output image: image_size/8ximage_size/8x128
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)     
        self.bn4 = nn.BatchNorm2d(128)

        # Input image_size/8ximage_size/8x64 image
        # 256 filters
        # 3x3 filter size 
        # Output image: image_size/8ximage_size/8x256
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)     
        self.bn5 = nn.BatchNorm2d(256)
            
        self.fc1 = nn.Linear(int((image_size / 8)) ** 2 * 256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)         
        return x