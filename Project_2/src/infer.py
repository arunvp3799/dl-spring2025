import pickle
import os
import logging
from typing import Optional, Dict, List, Any
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from peft import PeftModel, PeftConfig
from train import tokenize_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceConfig(BaseModel):
    """Configuration for inference parameters."""
    model_path: str = Field(..., description="Path to the trained model")
    test_data_path: str = Field(..., description="Path to the test data file")
    output_dir: str = Field(default="Results", description="Directory to save results")
    output_filename: str = Field(default="submission.csv", description="Output filename")
    max_length: int = Field(default=256, ge=1, description="Maximum sequence length")
    
    @field_validator('model_path', 'test_data_path')
    def validate_path_exists(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Path does not exist: {v}")
        return v
    
    @field_validator('output_dir')
    def validate_output_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

def inference(
    model_path: str, 
    test_data_path: str,
    output_dir: str = "Results",
    output_filename: str = "submission.csv",
    max_length: int = 256
) -> None:
    """
    Run inference using a trained model on test data.
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test data file
        output_dir: Directory to save results
        output_filename: Output filename
        max_length: Maximum sequence length for tokenization
    """
    try:
        # Validate configuration
        config = InferenceConfig(
            model_path=model_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            output_filename=output_filename,
            max_length=max_length
        )
        
        logger.info(f"Loading model from {config.model_path}")
        peft_config = PeftConfig.from_pretrained(config.model_path)
        base_model = RobertaForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path, 
            num_labels=4
        )
        model = PeftModel.from_pretrained(base_model, config.model_path)
        
        logger.info(f"Loading tokenizer")
        tokenizer = RobertaTokenizer.from_pretrained(peft_config.base_model_name_or_path)

        logger.info("Setting up classification pipeline")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        logger.info(f"Loading test data from {config.test_data_path}")
        with open(config.test_data_path, "rb") as f:
            test_data = pickle.load(f)

        logger.info("Tokenizing test data")
        try:
            test_data = test_data.map(
                lambda x: tokenize_function(x, tokenizer),
                batched=True,
                remove_columns=["text"],
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise

        logger.info("Running inference")
        predictions = classifier(test_data["text"], truncation=True, max_length=config.max_length)

        logger.info("Processing predictions")
        results = {
            "ID": [],
            "Label": []
        }
        
        for idx, pred in enumerate(predictions):
            label_id = int(pred['label'].split('_')[-1])
            results["Label"].append(label_id)
            results["ID"].append(idx)

        output_path = os.path.join(config.output_dir, config.output_filename)
        logger.info(f"Saving results to {output_path}")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--output_dir", type=str, default="Results", help="Directory to save results")
    parser.add_argument("--output_filename", type=str, default="submission.csv", help="Output filename")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")

    args = parser.parse_args()
    
    inference(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        max_length=args.max_length
    )