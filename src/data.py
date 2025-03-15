import os
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
import pickle
import numpy as np
import datasets

## Create Custom Dataset Class

train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])


test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])


## Write a function to create the data that applies the transforms as well like CIFAR format

def create_data() -> tuple:
    """
    Function to create the data from the given path
    Args:
    train_data_path : str : path of the train data
    valid_data_path : str : path of the valid data
    test_data_path : str : path of the test data

    Returns:
    Tuple : train data, valid data, test data
    """
    train_dataset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=train_transform)
    valid_dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=test_transform)
    return train_dataset, valid_dataset

if __name__ == "__main__":
    train_dataset, valid_dataset = create_data(train_data_path="../data/cifar-10-python/cifar-10-batches-py", valid_data_path="../data/cifar-10-python/cifar-10-batches-py/test_batch", test_data_path="")
    