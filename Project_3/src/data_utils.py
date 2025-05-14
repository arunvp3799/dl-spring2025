import numpy as np
import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, labels_json_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class folders.
            labels_json_path (string): Path to the JSON file containing class labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load labels from JSON file
        with open(labels_json_path, 'r') as f:
            self.labels_list = json.load(f)
            self.idx_to_label = {int(x.split(':')[0]): x.split(':')[1].strip() for x in self.labels_list}
        
        # Get all class folders and sort them
        self.folders = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create a mapping from folder to label
        self.folder_to_label = {}
        for i, folder in enumerate(self.folders):
            # Labels start from 401 in the JSON file
            label_idx = 401 + i
            self.folder_to_label[folder] = self.labels_list[i]
        
        # Get all image paths and their corresponding labels
        self.images = []
        self.labels = []
        
        for folder in self.folders:
            folder_path = os.path.join(root_dir, folder)
            label_idx = 401 + self.folders.index(folder)  # Calculate label index (401-500)
            
            for img_name in sorted(os.listdir(folder_path)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(folder_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class LoadedPerturbedDataset(Dataset):
    def __init__(self, perturbed_images, perturbed_labels):
        self.perturbed_images = perturbed_images
        self.perturbed_labels = perturbed_labels

    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        return self.perturbed_images[idx], self.perturbed_labels[idx]
