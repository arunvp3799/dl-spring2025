from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from data_utils import ImageFolderDataset, LoadedPerturbedDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
import random
import argparse
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, field_validator, Field

# Define configuration class with Pydantic
class RandomPatchedPGDConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to use (resnet34 or densenet121)")
    batch_size: int = Field(32, description="Batch size for dataloader")
    data_dir: str = Field("data/TestDataSet", description="Directory containing the dataset")
    labels_json: str = Field("data/TestDataSet/labels_list.json", description="Path to the labels JSON file")
    save_dir: str = Field("prediction_samples", description="Directory to save visualization results")
    epsilon: float = Field(0.5, description="Epsilon value for PGD attack")
    alpha: float = Field(0.005, description="Step size for PGD attack")
    num_iterations: int = Field(500, description="Number of iterations for PGD attack")
    num_samples: int = Field(5, description="Number of samples to visualize")
    
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
    
    @field_validator('epsilon')
    def validate_epsilon(cls, v):
        if v <= 0 or v > 1.0:
            raise ValueError(f"Epsilon should be between 0 and 1.0, got {v}")
        return v
    
    @field_validator('num_iterations')
    def validate_num_iterations(cls, v):
        if v <= 0:
            raise ValueError(f"Number of iterations must be positive, got {v}")
        return v

class RandomPatchedPGDAttack:
    def __init__(self, config: RandomPatchedPGDConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.model_name
        self.save_dir = os.path.join(config.save_dir, self.model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        try:
            self.model = self._load_model()
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
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
    
    def random_patched_pgd_attack(self):
        """
        Perform random patched PGD attack on the whole dataloader.
        Returns all perturbed images and labels.
        """
        self.model.eval()
        adv_images_all = []
        adv_labels_all = []
        original_images_all = []
        
        # Use tqdm for tracking progress
        for images, labels in tqdm(self.dataloader, desc=f"Performing random patched PGD attack on {self.model_name}"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            original_images = images.clone().detach()
            original_images_all.append(original_images.detach().cpu())
            perturbed_images = images.clone().detach().to(self.device)
            perturbed_images.requires_grad = True

            for _ in tqdm(range(self.config.num_iterations), desc="PGD iterations", leave=False):
                try:
                    outputs = self.model(perturbed_images)
                    loss = F.cross_entropy(outputs, labels)

                    self.model.zero_grad()
                    loss.backward()

                    # Get gradient
                    gradients = perturbed_images.grad
                    if gradients is None:
                        raise ValueError("Gradient is None. Make sure requires_grad=True on the input.")

                    # Randomized patched PGD step
                    patch_perturbed_images = []
                    for img_num, perturbed_image in enumerate(perturbed_images):
                        copied_image = perturbed_image.clone().detach()
                        original_perturbed_image = perturbed_image.clone().detach()
                        rand_x = random.randint(0, perturbed_image.shape[1] - 32)
                        rand_y = random.randint(0, perturbed_image.shape[2] - 32)
                        patch = perturbed_image[:, rand_x:rand_x+32, rand_y:rand_y+32] + self.config.alpha * gradients.sign()[img_num, :, rand_x:rand_x+32, rand_y:rand_y+32]
                        patch = torch.min(torch.max(patch, original_perturbed_image[:, rand_x:rand_x+32, rand_y:rand_y+32] - self.config.epsilon), 
                                        original_perturbed_image[:, rand_x:rand_x+32, rand_y:rand_y+32] + self.config.epsilon)
                        patch = torch.clamp(patch, original_images[img_num, :, rand_x:rand_x+32, rand_y:rand_y+32] - self.config.epsilon, 
                                        original_images[img_num, :, rand_x:rand_x+32, rand_y:rand_y+32] + self.config.epsilon)
                        copied_image[:, rand_x:rand_x+32, rand_y:rand_y+32] = patch
                        patch_perturbed_images.append(copied_image)

                    perturbed_images = torch.stack(patch_perturbed_images)
                    # Detach to stop accumulating gradient history, and re-enable grad
                    perturbed_images = perturbed_images.detach()
                    perturbed_images.requires_grad = True
                except Exception as e:
                    print(f"Error during PGD iteration: {str(e)}")
                    break

            adv_images_all.append(perturbed_images.detach().cpu())
            adv_labels_all.append(labels.detach().cpu())
            
        # Concatenate all batches
        adv_images_all = torch.cat(adv_images_all)
        adv_labels_all = torch.cat(adv_labels_all)
        original_images_all = torch.cat(original_images_all)

        return adv_images_all, adv_labels_all, original_images_all
    
    def evaluate_model_on_perturbed_images(self, perturbed_images, labels):
        """
        Evaluate the model on perturbed images and return accuracy metrics.
        """
        perturbed_dataset = LoadedPerturbedDataset(perturbed_images, labels)
        perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=16, shuffle=False)
        
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        all_predictions = []
        all_true_labels = []
        
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(perturbed_dataloader, desc=f"Evaluating {self.model_name} on perturbed images"):
                try:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    # Get top-5 predictions
                    top5_prob, top5_indices = torch.topk(outputs, k=5, dim=1)  # Shape: [batch_size, 5]

                    top1_preds = top5_indices[:, 0]
                    # Compare actual and predicted labels using string class names
                    for i in range(labels.size(0)):
                        true_label_idx = labels[i].item()
                        true_label_name = self.idx_to_label[true_label_idx]

                        top1_pred_idx = top1_preds[i].item()
                        top1_label_name = self.idx_to_label.get(top1_pred_idx, "")
                        
                        # Store for visualization
                        all_predictions.append(top1_pred_idx)
                        all_true_labels.append(true_label_idx)

                        if top1_label_name == true_label_name:
                            top1_correct += 1

                        # Check if true label is in top-5 predictions
                        top5_pred_indices = top5_indices[i].tolist()
                        top5_label_names = [self.idx_to_label.get(idx, "") for idx in top5_pred_indices]

                        if true_label_name in top5_label_names:
                            top5_correct += 1

                    total_samples += labels.size(0)
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue
        
        # Final accuracy computation
        top1_acc = top1_correct / total_samples * 100 if total_samples > 0 else 0
        top5_acc = top5_correct / total_samples * 100 if total_samples > 0 else 0
        
        print(f"Total samples: {total_samples}")
        print(f"Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
        
        return all_predictions, all_true_labels, top1_acc, top5_acc
    
    def visualize_attack_results(self, original_images, perturbed_images, labels, predictions):
        """
        Visualize the attack results and save to file.
        """
        # Create directory for saving visualization
        os.makedirs(self.save_dir, exist_ok=True)
        
        # For denormalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        def denormalize(img):
            return img * std + mean
        
        # Select random indices for visualization
        num_samples = min(self.config.num_samples, len(labels))
        indices = random.sample(range(len(labels)), num_samples)
        
        # Create figure for visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 8))
        
        # Create a text file to log predictions
        log_file = os.path.join(self.save_dir, f"{self.model_name}_random_patched_pgd_prediction_log.txt")
        with open(log_file, "w") as f:
            f.write(f"{self.model_name} Random Patched PGD Attack Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, idx in enumerate(indices):
                try:
                    label = labels[idx].item()
                    pred = predictions[idx]
                    
                    # Get original and perturbed images
                    orig_img = denormalize(original_images[idx]).clamp(0, 1)
                    pert_img = denormalize(perturbed_images[idx]).clamp(0, 1)
                    
                    # Get label names
                    true_label_name = self.idx_to_label.get(label, "Unknown")
                    pred_label_name = self.idx_to_label.get(pred, "Unknown")
                    
                    # Log predictions to file
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  True Label ID: {label} ({true_label_name})\n")
                    f.write(f"  Perturbed Image Prediction ID: {pred} ({pred_label_name})\n")
                    f.write("\n" + "-"*50 + "\n\n")
                    
                    # Plot original image
                    axes[0, i].imshow(orig_img.permute(1, 2, 0).cpu().numpy())
                    axes[0, i].set_title(f"Original\nTrue: {true_label_name} (ID: {label})")
                    axes[0, i].axis('off')
                    
                    # Plot perturbed image
                    axes[1, i].imshow(pert_img.permute(1, 2, 0).cpu().numpy())
                    axes[1, i].set_title(f"Perturbed\nTrue: {true_label_name} (ID: {label})\nPred: {pred_label_name} (ID: {pred})")
                    axes[1, i].axis('off')
                except Exception as e:
                    print(f"Error visualizing sample {i}: {str(e)}")
                    continue
        
        plt.tight_layout()
        img_file = os.path.join(self.save_dir, f"{self.model_name}_random_patched_pgd_attack_comparison.png")
        plt.savefig(img_file)
        plt.close(fig)
        
        print(f"Random patched PGD attack visualization saved to {img_file}")
        print(f"Prediction details logged to {log_file}")
    
    def run(self):
        """
        Run the full attack pipeline.
        """
        try:
            # Perform the attack
            perturbed_images, perturbed_labels, original_images = self.random_patched_pgd_attack()
            
            # Evaluate the model on perturbed images
            predictions, true_labels, top1_acc, top5_acc = self.evaluate_model_on_perturbed_images(
                perturbed_images, perturbed_labels
            )
            
            # Visualize the results
            self.visualize_attack_results(
                original_images, perturbed_images, perturbed_labels, predictions
            )
            
            return {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "model_name": self.model_name
            }
        except Exception as e:
            print(f"Error running attack pipeline: {str(e)}")
            return None

def parse_args():
    parser = argparse.ArgumentParser(description="Random Patched PGD Attack")
    parser.add_argument("--model", type=str, required=True, choices=["resnet34", "densenet121"],
                        help="Model to use (resnet34 or densenet121)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Create config from args, using only model_name from arguments and defaults for the rest
        config = RandomPatchedPGDConfig(
            model_name=args.model
        )
        
        # Initialize and run the attack
        attack = RandomPatchedPGDAttack(config)
        results = attack.run()
        
        if results:
            print(f"Attack completed successfully on {results['model_name']}")
            print(f"Final Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
            print(f"Final Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    except Exception as e:
        print(f"Error: {str(e)}")