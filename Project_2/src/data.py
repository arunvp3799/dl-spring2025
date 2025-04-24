from datasets import load_dataset, concatenate_datasets
from augment import augment_text, AugmentationConfig
from typing import Dict, Tuple, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuration for data loading and augmentation."""
    num_samples: int = Field(default=40000, ge=1, description="Number of samples to use")
    dataset_name: str = Field(default="ag_news", description="Name of the dataset to load")
    augmentation_config: Optional[AugmentationConfig] = Field(
        default=None, 
        description="Configuration for text augmentation"
    )
    
    @field_validator('num_samples')
    def validate_num_samples(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_samples must be positive")
        return v


def get_augmented_agnews_data(
    num_samples: int, 
    augmentation_config: Optional[AugmentationConfig] = None
) -> Tuple[Any, Any, Dict[int, str], Dict[str, int], List[str]]:
    """
    Load and augment AG News dataset.
    
    Args:
        num_samples: Number of samples to use per class
        augmentation_config: Configuration for text augmentation
        
    Returns:
        Tuple containing:
        - train_dataset: Augmented training dataset
        - test_data: Test dataset
        - id2label: Mapping from label IDs to label names
        - label2id: Mapping from label names to label IDs
        - class_names: List of class names
    """
    try:
        # Validate configuration
        config = DataConfig(
            num_samples=num_samples,
            augmentation_config=augmentation_config
        )
        
        logger.info(f"Loading dataset: {config.dataset_name}")
        agnews = load_dataset(config.dataset_name)
        
        class_names = agnews["train"].features["label"].names
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}

        test_data = agnews["test"]
        train_data = agnews["train"]

        # Sample balanced data from each class
        labels = sorted(set(train_data["label"]))
        per_label = config.num_samples // len(labels)
        indices_by_label = {lab: [] for lab in labels}

        for i, lab in enumerate(train_data["label"]):
            indices_by_label[lab].append(i)
        
        sampled_indices = []
        for lab in labels:
            if len(indices_by_label[lab]) < per_label:
                logger.warning(f"Not enough samples for label {lab}. Using all available {len(indices_by_label[lab])} samples.")
                sampled_indices += indices_by_label[lab]
            else:
                sampled_indices += random.sample(indices_by_label[lab], per_label)
        
        subset = train_data.select(sampled_indices)
        logger.info(f"Selected {len(subset)} samples for augmentation")

        # Define augmentation function
        def perturb(ex):
            try:
                return {"text": augment_text(ex["text"], config=config.augmentation_config)}
            except Exception as e:
                logger.error(f"Error augmenting text: {str(e)}")
                return {"text": ex["text"]}  # Return original text if augmentation fails

        # Apply augmentation
        logger.info("Applying text augmentation")
        perturbed_subset = subset.map(perturb)

        # Combine original and augmented data
        train_dataset = concatenate_datasets([train_data, perturbed_subset])
        logger.info(f"Final training dataset size: {len(train_dataset)}")

        return train_dataset, test_data, id2label, label2id, class_names
        
    except Exception as e:
        logger.error(f"Error in get_augmented_agnews_data: {str(e)}")
        raise
