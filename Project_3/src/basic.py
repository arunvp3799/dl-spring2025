from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torchvision.models as models
from data_utils import ImageFolderDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import argparse
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, field_validator, Field

class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to use (resnet34 or densenet121)")
    batch_size: int = Field(32, description="Batch size for dataloader")
    data_dir: str = Field("data/TestDataSet", description="Directory containing the dataset")
    labels_json: str = Field("data/TestDataSet/labels_list.json", description="Path to the labels JSON file")
    save_dir: str = Field("prediction_samples", description="Directory to save visualization results")
    
    @field_validator('model_name')
    def validate_model_name(cls, v):
        if v not in ['resnet34', 'densenet121']:
            raise ValueError(f"Model name must be 'resnet34' or 'densenet121', got {v}")
        return v
    
    @field_validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got {v}")
        return v

class ImageClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.model_name
        self.save_dir = os.path.join(config.save_dir, self.model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Display transform to convert tensor back to image
        self.display_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            ),
        ])
        
        # Load dataset
        try:
            self.dataset = ImageFolderDataset(
                root_dir=config.data_dir, 
                labels_json_path=config.labels_json, 
                transform=self.transform
            )
            self.dataloader = DataLoader(
                self.dataset, 
                batch_size=config.batch_size, 
                shuffle=True
            )
            self.idx_to_label = self.dataset.idx_to_label
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
    
    def _load_model(self):
        try:
            if self.model_name == 'resnet34':
                return models.resnet34(weights='IMAGENET1K_V1')
            elif self.model_name == 'densenet121':
                return models.densenet121(weights='IMAGENET1K_V1')
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def evaluate(self):
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        visualized = False
        
        try:
            with torch.no_grad():
                for images, labels in self.dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    
                    # Get top-5 predictions
                    probabilities = F.softmax(outputs, dim=1)
                    top5_prob, top5_indices = torch.topk(probabilities, k=5, dim=1)
                    
                    top1_preds = top5_indices[:, 0]
                    
                    # Compare actual and predicted labels using string class names
                    for i in range(labels.size(0)):
                        true_label_idx = labels[i].item()
                        true_label_name = self.idx_to_label[true_label_idx]
                        
                        top1_pred_idx = top1_preds[i].item()
                        top1_label_name = self.idx_to_label.get(top1_pred_idx, "")
                        
                        if top1_label_name == true_label_name:
                            top1_correct += 1
                        
                        # Check if true label is in top-5 predictions
                        top5_pred_indices = top5_indices[i].tolist()
                        top5_label_names = [self.idx_to_label.get(idx, "") for idx in top5_pred_indices]
                        
                        if true_label_name in top5_label_names:
                            top5_correct += 1
                    
                    # Visualize 3 sample predictions if not done yet
                    if not visualized and images.size(0) >= 3:
                        self._visualize_predictions(images, labels, top1_preds, top5_prob)
                        visualized = True
                    
                    total_samples += labels.size(0)
            
            # Final accuracy computation
            top1_acc = top1_correct / total_samples * 100
            top5_acc = top5_correct / total_samples * 100
            
            print(f"Model: {self.model_name}")
            print(f"Total samples: {total_samples}")
            print(f"Top-1 Accuracy: {top1_acc:.2f}%")
            print(f"Top-5 Accuracy: {top5_acc:.2f}%")
            
            # Save results to a log file
            with open(os.path.join(self.save_dir, f"{self.model_name}_results.txt"), "w") as f:
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Total samples: {total_samples}\n")
                f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
                f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
            
            return {
                "model": self.model_name,
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "total_samples": total_samples
            }
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def _visualize_predictions(self, images, labels, top1_preds, top5_prob):
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for j in range(3):
                # Convert tensor to displayable image
                img_display = self.display_transform(images[j].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
                
                # Get true and predicted labels
                true_idx = labels[j].item()
                pred_idx = top1_preds[j].item()
                true_name = self.idx_to_label[true_idx]
                pred_name = self.idx_to_label.get(pred_idx, "Unknown")
                confidence = top5_prob[j, 0].item() * 100  # Convert to percentage
                
                # Plot the image
                axes[j].imshow(img_display)
                # Use a smaller font size and truncate long class names to prevent cropping
                true_name_short = true_name[:15] + "..." if len(true_name) > 15 else true_name
                pred_name_short = pred_name[:15] + "..." if len(pred_name) > 15 else pred_name
                axes[j].set_title(f"True: {true_name_short}\nPred: {pred_name_short}\nConf: {confidence:.2f}%", fontsize=8)
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{self.model_name}_prediction_samples.png"), bbox_inches='tight')
            plt.close(fig)
            print(f"Sample predictions saved to {self.save_dir} directory")
        except Exception as e:
            print(f"Error during visualization: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification with PyTorch")
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34", "densenet121"],
                        help="Model architecture to use (resnet34 or densenet121)")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        
        # Create config from args, using defaults for other parameters
        config = ModelConfig(
            model_name=args.model
        )
        
        # Initialize and run classifier
        classifier = ImageClassifier(config)
        results = classifier.evaluate()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")