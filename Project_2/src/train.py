from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model as create_peft_model, TaskType
from data import get_augmented_agnews_data
from preprocess import preprocess, mask_text
import torch
from evaluate import load
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pydantic import BaseModel, Field, validator, ValidationError
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the accuracy metric using the evaluate library
accuracy_metric = load("accuracy")

# Pydantic models for validation
class ModelConfig(BaseModel):
    model_id: str = Field(default="roberta-base", description="Pretrained model identifier")
    num_labels: int = Field(default=4, ge=1, description="Number of classification labels")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="Device for training")
    
    @validator('device')
    def validate_device(cls, v):
        if v == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return v

class TrainingConfig(BaseModel):
    num_samples: int = Field(default=40000, ge=1, description="Number of samples to use")
    train_batch_size: int = Field(default=64, ge=1, description="Training batch size")
    test_batch_size: int = Field(default=32, ge=1, description="Testing batch size")
    max_steps: int = Field(default=1000, ge=1, description="Maximum training steps")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    logging_steps: int = Field(default=50, ge=1, description="Steps between logging")
    eval_steps: int = Field(default=50, ge=1, description="Steps between evaluation")
    save_steps: int = Field(default=50, ge=1, description="Steps between saving")
    output_dir: str = Field(default="../Models", description="Output directory for models")
    run_name: str = Field(default="run_1", description="Name for the training run")

class LoRAConfig(BaseModel):
    r: int = Field(default=4, ge=1, description="LoRA attention dimension")
    lora_alpha: int = Field(default=8, ge=1, description="LoRA alpha parameter")
    target_modules: List[str] = Field(
        default=["self_attn.value", "self_attn.query", "self_attn.key", "output.dense"],
        description="Target modules for LoRA"
    )
    lora_dropout: float = Field(default=0.1, ge=0, le=1, description="LoRA dropout rate")
    bias: str = Field(default='none', description="LoRA bias type")
    task_type: TaskType = Field(default=TaskType.SEQ_CLS, description="Task type for LoRA")
    use_dora: bool = Field(default=True, description="Whether to use DoRA")


def tokenize_function(examples: Dict[str, List], tokenizer: RobertaTokenizer) -> Dict[str, List]:
    """
    Tokenize and preprocess text examples.
    
    Args:
        examples: Dictionary containing text examples
        tokenizer: Tokenizer to use for tokenization
        
    Returns:
        Dictionary with tokenized inputs
    """
    try:
        examples["text"] = [preprocess(text) for text in examples["text"]]
        examples["text"] = [mask_text(text) for text in examples["text"]]
        examples["text"] = [text.replace("\n", " ") for text in examples["text"]]

        tokenizer_resp = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        examples["input_ids"] = tokenizer_resp["input_ids"]
        examples["attention_mask"] = tokenizer_resp["attention_mask"]
        return examples
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        raise


def load_tokenizer(model_id: str) -> RobertaTokenizer:
    """
    Load the tokenizer for the specified model ID.
    
    Args:
        model_id: The model ID for loading the tokenizer.
    
    Returns:
        The loaded tokenizer.
    """
    try:
        logger.info(f"Loading tokenizer for model: {model_id}")
        return RobertaTokenizer.from_pretrained(model_id)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise


def load_model(model_id: str, num_labels: int, id2label: Dict[int, str]) -> RobertaForSequenceClassification:
    """
    Load the model for sequence classification.
    
    Args:
        model_id: The model ID for loading the model.
        num_labels: The number of labels for classification.
        id2label: Mapping from label IDs to label names.
    
    Returns:
        The loaded model.
    """
    try:
        logger.info(f"Loading model: {model_id} with {num_labels} labels")
        return RobertaForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def apply_peft_model(model: RobertaForSequenceClassification, config: LoraConfig) -> Any:
    """
    Apply Parameter-Efficient Fine-Tuning (PEFT) to the model.
    
    Args:
        model: Base model to apply PEFT to
        config: LoRA configuration
        
    Returns:
        PEFT model
    """
    try:
        logger.info("Applying PEFT to model")
        model = create_peft_model(model, config)
        trainable_params = model.num_parameters()
        logger.info(f"Trainable parameters: {trainable_params}")
        return model
    except Exception as e:
        logger.error(f"Failed to apply PEFT: {str(e)}")
        raise


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute evaluation metrics from predictions and labels.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        Dictionary of metric names and values
    """
    try:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return {"accuracy": 0.0}


def train(
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    lora_config: Optional[LoRAConfig] = None
) -> None:
    """
    Main training function.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        lora_config: LoRA configuration
    """
    try:
        # Initialize configurations with defaults if not provided
        if model_config is None:
            model_config = ModelConfig()
        if training_config is None:
            training_config = TrainingConfig()
        if lora_config is None:
            lora_config = LoRAConfig()
            
        logger.info("Starting training with configurations:")
        logger.info(f"Model config: {model_config.dict()}")
        logger.info(f"Training config: {training_config.dict()}")
        logger.info(f"LoRA config: {lora_config.dict()}")

        # Define the LoRA configuration
        lora_config_obj = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            use_dora=lora_config.use_dora
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(training_config.output_dir, exist_ok=True)
        
        # Load the augmented dataset
        logger.info(f"Loading dataset with {training_config.num_samples} samples")
        train_dataset, test_dataset, id2label, label2id, class_names = get_augmented_agnews_data(training_config.num_samples)
        logger.info(f"Dataset loaded with {len(train_dataset)} training and {len(test_dataset)} test examples")

        # Load the tokenizer and model
        tokenizer = load_tokenizer(model_config.model_id)
        model = load_model(model_config.model_id, model_config.num_labels, id2label)
        model = apply_peft_model(model, lora_config_obj)
        model = model.to(model_config.device)

        # Tokenize the dataset
        logger.info("Tokenizing training dataset")
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text"],
        )

        logger.info("Tokenizing test dataset")
        test_dataset = test_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text"],
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        trainer_args = TrainingArguments(
            output_dir=training_config.output_dir,
            overwrite_output_dir=True,
            max_steps=training_config.max_steps,
            logging_strategy="steps",
            logging_steps=training_config.logging_steps,
            eval_strategy="steps",
            eval_steps=training_config.eval_steps,
            save_steps=training_config.save_steps,
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=training_config.train_batch_size,
            per_device_eval_batch_size=training_config.test_batch_size,
            weight_decay=training_config.weight_decay,
            optim="adamw_torch",
            label_names=["label"],
            report_to="wandb",
            run_name=training_config.run_name
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        logger.info("Starting training")
        trainer.train()
        
        logger.info("Evaluating model")
        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"Evaluation results: {eval_results}")

        logger.info(f"Training completed successfully")
        
        model_save_path = os.path.join(training_config.output_dir, "final_model")
        logger.info(f"Saving model to {model_save_path}")
        trainer.save_model(model_save_path)
        logger.info("Model saved successfully")
        
    except ValidationError as e:
        logger.error(f"Configuration validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")