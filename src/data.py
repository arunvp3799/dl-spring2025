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

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label  

def create_data(train_data_path:str="", valid_data_path:str="") -> tuple:
    """
    Function to create the data from the given path
    Args:
    train_data_path : str : path of the train data
    valid_data_path : str : path of the valid data
    test_data_path : str : path of the test data

    Returns:
    Tuple : train data, valid data, test data
    """
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Train data path {train_data_path} does not exist")
    if not os.path.exists(valid_data_path):
        raise FileNotFoundError(f"Valid data path {valid_data_path} does not exist")
    
    try:
        all_train_data = []
        all_train_labels = []
        for i in range(1,6):
            train_batch_file = os.path.join(train_data_path, f"data_batch_{i}")
            with open(train_batch_file, 'rb') as file:
                data = pickle.load(file, encoding='bytes')
                all_train_data.extend(data[b'data'])
                all_train_labels.extend(data[b'labels'])

        all_valid_data = []
        all_valid_labels = []
        with open(valid_data_path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            all_valid_data.extend(data[b'data'])
            all_valid_labels.extend(data[b'labels'])
        
        all_train_data = np.array(all_train_data)
        all_train_labels = np.array(all_train_labels)
        all_valid_data = np.array(all_valid_data)
        all_valid_labels = np.array(all_valid_labels)

        all_train_data = all_train_data.reshape(-1, 3, 32, 32)
        all_valid_data = all_valid_data.reshape(-1, 3, 32, 32)

        all_train_data = torch.tensor(all_train_data, dtype=torch.float32)
        all_train_labels = torch.tensor(all_train_labels, dtype=torch.long)
        all_valid_data = torch.tensor(all_valid_data, dtype=torch.float32)
        all_valid_labels = torch.tensor(all_valid_labels, dtype=torch.long)

        train_dataset = CIFAR10Dataset(all_train_data, all_train_labels, transform=train_transform)
        valid_dataset = CIFAR10Dataset(all_valid_data, all_valid_labels, transform=test_transform)
        
    except Exception as e:
        raise Exception(f"Error in creating the data from the given path: {e}")

    return train_dataset, valid_dataset

if __name__ == "__main__":
    train_dataset, valid_dataset = create_data(train_data_path="../data/cifar-10-python/cifar-10-batches-py", valid_data_path="../data/cifar-10-python/cifar-10-batches-py/test_batch", test_data_path="")
    