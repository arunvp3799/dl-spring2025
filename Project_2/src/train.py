from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from data import get_augmented_agnews_data
from preprocess import preprocess, mask_text
import torch
from evaluate import load
import numpy as np

# Load the accuracy metric using the evaluate library
accuracy_metric = load("accuracy")

def tokenize_function(examples, tokenizer):
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


def load_tokenizer(model_id):
    """
    Load the tokenizer for the specified model ID.
    
    Args:
        model_id (str): The model ID for loading the tokenizer.
    
    Returns:
        RobertaTokenizer: The loaded tokenizer.
    """
    return RobertaTokenizer.from_pretrained(model_id)

def load_model(model_id, num_labels, id2label):
    """
    Load the model for sequence classification.
    
    Args:
        model_id (str): The model ID for loading the model.
        num_labels (int): The number of labels for classification.
        id2label (dict): Mapping from label IDs to label names.
    
    Returns:
        RobertaForSequenceClassification: The loaded model.
    """
    return RobertaForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
    )

def get_peft_model(model, config):
    
    model = get_peft_model(model, config)
    print(f"Trainable parameters: {model.num_parameters()}")
    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = 40000
    model_id = "roberta-base"
    num_labels = 4
    train_batch_size = 64
    test_batch_size = 32

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["self_attn.value", "self_attn.query", "self_attn.key", "output.dense"],
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.SEQ_CLS,
        use_dora=True
    )
    
    # Load the augmented dataset
    train_dataset, test_dataset, id2label, label2id, class_names = get_augmented_agnews_data(num_samples)

    # Load the tokenizer and model
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, num_labels, id2label)
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # Tokenize the dataset
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainer_args = TrainingArguments(
        output_dir="../Models",
        overwrite_output_dir=True,
        max_steps=1000,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        learning_rate=2e-4,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        weight_decay=0.01,
        optim="adamw_torch",
        label_names=["label"],
        report_to="wandb",
        run_name="run_1"
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_dataset)

    print(f"Training Done, Plotting losses and accuracies ...")

    trainer.save_model("../Models")

if __name__ == "__main__":
    train()