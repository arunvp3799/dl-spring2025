from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from data_utils import ImageFolderDataset, LoadedPerturbedDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from torchvision import transforms
import argparse
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, field_validator, Field

class FGSMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to use (resnet34 or densenet121)")
    batch_size: int = Field(32, description="Batch size for dataloader")
    data_dir: str = Field("data/TestDataSet", description="Directory containing the dataset")
    labels_json: str = Field("data/TestDataSet/labels_list.json", description="Path to the labels JSON file")
    save_dir: str = Field("prediction_samples", description="Directory to save visualization results")
    epsilon: float = Field(0.02, description="Epsilon value for FGSM attack")
    num_samples: int = Field(3, description="Number of samples to visualize")
    
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
        if v <= 0 or v > 0.5:
            raise ValueError(f"Epsilon should be between 0 and 0.5, got {v}")
        return v

class FGSMAttack:
    def __init__(self, config: FGSMConfig):
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
    
    def generate_adversarial_examples(self):
        """Generate adversarial examples using FGSM attack"""
        perturbed_images = []
        perturbed_image_labels = []
        original_images = []  # Store original images for comparison
        
        try:
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Store original images before perturbation
                original_images.extend(images.clone().detach().cpu())
                
                images.requires_grad = True
                outputs = self.model(images)

                # Use cross-entropy loss for the entire batch
                loss = F.cross_entropy(outputs, labels)
                self.model.zero_grad()
                loss.backward()  # Compute gradients w.r.t. input

                # Now gradients w.r.t. input images are in images.grad
                input_gradients = images.grad.clone().detach()

                perturbed_inputs = images + self.config.epsilon * torch.sign(input_gradients)
                perturbed_images.extend(perturbed_inputs.detach().cpu())
                perturbed_image_labels.extend(labels.cpu())
                
            return original_images, perturbed_images, perturbed_image_labels
        except Exception as e:
            raise RuntimeError(f"Error generating adversarial examples: {str(e)}")
    
    def evaluate_adversarial_examples(self, perturbed_images, perturbed_image_labels):
        """Evaluate model performance on adversarial examples"""
        perturbed_dataset = LoadedPerturbedDataset(perturbed_images, perturbed_image_labels)
        perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=self.config.batch_size, shuffle=True)

        top1_correct_fgsm = 0
        top5_correct_fgsm = 0
        total_samples_perturbed = 0
        
        try:
            with torch.no_grad():
                for images, labels in perturbed_dataloader:
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

                        if top1_label_name == true_label_name:
                            top1_correct_fgsm += 1

                        # Check if true label is in top-5 predictions
                        top5_pred_indices = top5_indices[i].tolist()
                        top5_label_names = [self.idx_to_label.get(idx, "") for idx in top5_pred_indices]

                        if true_label_name in top5_label_names:
                            top5_correct_fgsm += 1

                    total_samples_perturbed += labels.size(0)

            # Final accuracy computation
            top1_acc_fgsm = top1_correct_fgsm / total_samples_perturbed * 100
            top5_acc_fgsm = top5_correct_fgsm / total_samples_perturbed * 100

            print(f"Model: {self.model_name}")
            print(f"Total samples: {total_samples_perturbed}")
            print(f"Top-1 Accuracy: {top1_acc_fgsm:.2f}%")
            print(f"Top-5 Accuracy: {top5_acc_fgsm:.2f}%")
            
            # Save results to a log file
            with open(os.path.join(self.save_dir, f"{self.model_name}_fgsm_results.txt"), "w") as f:
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Total samples: {total_samples_perturbed}\n")
                f.write(f"Top-1 Accuracy: {top1_acc_fgsm:.2f}%\n")
                f.write(f"Top-5 Accuracy: {top5_acc_fgsm:.2f}%\n")
            
            return {
                "model": self.model_name,
                "top1_accuracy": top1_acc_fgsm,
                "top5_accuracy": top5_acc_fgsm,
                "total_samples": total_samples_perturbed
            }
        except Exception as e:
            raise RuntimeError(f"Error evaluating adversarial examples: {str(e)}")
    
    def visualize_attack_results(self, original_images, perturbed_images, labels):
        """Visualize original and perturbed image pairs with predictions"""
        try:
            # Select random indices
            indices = np.random.choice(len(original_images), self.config.num_samples, replace=False)
            
            # Create a figure with 2 rows (original and perturbed) and num_samples columns
            fig, axes = plt.subplots(2, self.config.num_samples, figsize=(15, 8))
            
            # Helper function to get label name with ID
            def get_label_with_id(idx):
                if idx in self.idx_to_label:
                    # Handle the case where the label might not have a colon
                    label_text = self.idx_to_label[idx]
                    if ':' in label_text:
                        return f"{idx}: {label_text.split(':')[1]}"
                    else:
                        return f"{idx}: {label_text}"
                else:
                    return f"{idx}: Unknown"
            
            # Create a log file for predictions
            log_file = os.path.join(self.save_dir, f"{self.model_name}_fgsm_prediction_log.txt")
            with open(log_file, "w") as f:
                f.write(f"{self.model_name} FGSM Attack Sample Predictions Log\n")
                f.write("=============================================\n\n")
                
                for i, idx in enumerate(indices):
                    # Get original image, perturbed image, and label
                    orig_img = original_images[idx]
                    pert_img = perturbed_images[idx]
                    label = labels[idx].item()
                    true_label_name = self.idx_to_label.get(label, f"Unknown")
                    
                    # Get predictions for original image
                    with torch.no_grad():
                        orig_output = self.model(orig_img.unsqueeze(0).to(self.device))
                        orig_top5_prob, orig_top5_indices = torch.topk(orig_output, k=5, dim=1)
                        orig_pred_idx = orig_top5_indices[0, 0].item()
                        orig_top5_ids = orig_top5_indices[0].tolist()
                        orig_top5_with_labels = [get_label_with_id(idx_val) for idx_val in orig_top5_ids]
                    
                    # Get predictions for perturbed image
                    with torch.no_grad():
                        pert_output = self.model(pert_img.unsqueeze(0).to(self.device))
                        pert_top5_prob, pert_top5_indices = torch.topk(pert_output, k=5, dim=1)
                        pert_pred_idx = pert_top5_indices[0, 0].item()
                        pert_top5_ids = pert_top5_indices[0].tolist()
                        pert_top5_with_labels = [get_label_with_id(idx_val) for idx_val in pert_top5_ids]
                    
                    # Log predictions to file
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  True Label ID: {label} ({true_label_name})\n")
                    f.write(f"  Original Image:\n")
                    f.write(f"    Top-1 ID: {orig_pred_idx} ({self.idx_to_label.get(orig_pred_idx, 'Unknown')})\n")
                    f.write(f"    Top-5 IDs with Labels: {orig_top5_with_labels}\n")
                    f.write(f"  Perturbed Image:\n")
                    f.write(f"    Top-1 ID: {pert_pred_idx} ({self.idx_to_label.get(pert_pred_idx, 'Unknown')})\n")
                    f.write(f"    Top-5 IDs with Labels: {pert_top5_with_labels}\n")
                    f.write("\n" + "-"*50 + "\n\n")
                    
                    # Convert tensors to displayable images
                    orig_img_display = self.display_transform(orig_img).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    pert_img_display = self.display_transform(pert_img).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    
                    # Get display labels with IDs for visualization - safely handle potential format issues
                    try:
                        orig_label_parts = get_label_with_id(orig_pred_idx).split(': ', 1)
                        orig_display = orig_label_parts[1] if len(orig_label_parts) > 1 else orig_label_parts[0]
                        
                        pert_label_parts = get_label_with_id(pert_pred_idx).split(': ', 1)
                        pert_display = pert_label_parts[1] if len(pert_label_parts) > 1 else pert_label_parts[0]
                        
                        true_label_parts = get_label_with_id(label).split(': ', 1)
                        true_display = true_label_parts[1] if len(true_label_parts) > 1 else true_label_parts[0]
                    except Exception as e:
                        # Fallback to simpler display if any issues
                        orig_display = self.idx_to_label.get(orig_pred_idx, "Unknown")
                        pert_display = self.idx_to_label.get(pert_pred_idx, "Unknown")
                        true_display = self.idx_to_label.get(label, "Unknown")
                    
                    # Plot original image
                    axes[0, i].imshow(orig_img_display)
                    axes[0, i].set_title(f"Original\nTrue: {true_display}\nPred: {orig_display} (ID: {orig_pred_idx})")
                    axes[0, i].axis('off')
                    
                    # Plot perturbed image
                    axes[1, i].imshow(pert_img_display)
                    axes[1, i].set_title(f"Perturbed\nTrue: {true_display}\nPred: {pert_display} (ID: {pert_pred_idx})")
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{self.model_name}_fgsm_attack_comparison.png"))
            plt.close(fig)
            
            print(f"FGSM attack visualization saved to {self.save_dir}/{self.model_name}_fgsm_attack_comparison.png")
            print(f"Prediction details logged to {self.save_dir}/{self.model_name}_fgsm_prediction_log.txt")
        except Exception as e:
            raise RuntimeError(f"Error visualizing attack results: {str(e)}")
    
    def verify_perturbation_bounds(self, original_images, perturbed_images):
        """Verify that perturbations are within the epsilon bound"""
        valid_perturbations = 0
        total_samples = min(len(original_images), len(perturbed_images))
        
        try:
            for i in range(total_samples):
                perturbed_image = perturbed_images[i].to(self.device)
                original_image = original_images[i].to(self.device)

                # Calculate the difference between perturbed and original image
                difference = perturbed_image - original_image
                # Calculate the max absolute difference
                max_diff = difference.abs().max().item()
                # Check if the difference is within epsilon
                tolerance = 1e-5
                is_within_epsilon = max_diff <= (self.config.epsilon + tolerance)
                if is_within_epsilon:
                    valid_perturbations += 1

            print(f"Model: {self.model_name}")
            print(f"Total samples checked: {total_samples}")
            print(f"Valid perturbations (within epsilon={self.config.epsilon}): {valid_perturbations}")
            print(f"Percentage valid: {(valid_perturbations/total_samples)*100:.2f}%")
            
            # Save results to a log file
            with open(os.path.join(self.save_dir, f"{self.model_name}_fgsm_verification.txt"), "w") as f:
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Epsilon: {self.config.epsilon}\n")
                f.write(f"Total samples checked: {total_samples}\n")
                f.write(f"Valid perturbations: {valid_perturbations}\n")
                f.write(f"Percentage valid: {(valid_perturbations/total_samples)*100:.2f}%\n")
            
            return {
                "model": self.model_name,
                "epsilon": self.config.epsilon,
                "total_samples": total_samples,
                "valid_perturbations": valid_perturbations,
                "percentage_valid": (valid_perturbations/total_samples)*100
            }
        except Exception as e:
            raise RuntimeError(f"Error verifying perturbation bounds: {str(e)}")
    
    def run(self):
        """Run the complete FGSM attack pipeline"""
        try:
            print(f"Running FGSM attack with {self.model_name} model...")
            original_images, perturbed_images, perturbed_labels = self.generate_adversarial_examples()
            results = self.evaluate_adversarial_examples(perturbed_images, perturbed_labels)
            self.visualize_attack_results(original_images, perturbed_images, perturbed_labels)
            verification = self.verify_perturbation_bounds(original_images, perturbed_images)
            return {
                "results": results,
                "verification": verification
            }
        except Exception as e:
            print(f"Error running FGSM attack: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='FGSM Attack on Image Classifiers')
    parser.add_argument('--model', type=str, required=True, choices=['resnet34', 'densenet121'],
                        help='Model to use (resnet34 or densenet121)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        config = FGSMConfig(
            model_name=args.model
        )
        
        fgsm_attack = FGSMAttack(config)
        results = fgsm_attack.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
