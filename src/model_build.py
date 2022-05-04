"""
CS5330 - Pattern Recognition & Computer Vision
Project Title: Face detection and filter application
April 18, 2022
Team: Sida Zhang, Xichen Liu, Xiang Wang, Hongyu Wan

Description: Structure of the model network and its classes.
"""
__author__ = "Sida Zhang, Hongyu Wan, Xiang Wang, Xichen Liu"

import os
import csv
import cv2
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

torch.manual_seed(888)


class MyNetwork(nn.Module):
    """
    Build CNN
    """

    def __init__(self, conv_filter = 5, dropout_rate = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, (conv_filter, conv_filter))
        self.conv2 = nn.Conv2d(10, 20, (conv_filter, conv_filter))
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.flat1 = nn.Flatten()
        outer = conv_filter // 2
        size = ((200 - 2 * outer) // 2 - 2 * outer) // 2
        self.fc1 = nn.Linear(20 * size * size, 50)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(x, (2, 2)))
        # A convolution layer with 20 5x5 filters
        x = self.conv2(x)
        # A dropout layer with a 0.5 dropout rate (50%)
        x = self.conv2_drop(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(x, (2, 2)))
        # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the
        # output
        x = F.relu(self.fc1(self.flat1(x)))

        return x


class CustomizedDataset(Dataset):
    """
    Generate a dataset
    """

    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def generate_csv(img_dir, csv_name):
    """
    Generate the csv file form a given image directory

    :param img_dir:     image directory
    :param csv_name:    name of csv file
    :return
    """
    dir_path = '../data/' + img_dir
    csv_path = '../data/' + csv_name
    with open(csv_path, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.writer(f)
        header = ['Filename', 'Label']
        writer.writerow(header)
        for filename in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, filename))
            img = cv2.resize(img, (200, 200))
            cv2.imwrite(os.path.join(dir_path, filename), img)
            writer.writerow([filename, os.path.join(dir_path, filename)])
