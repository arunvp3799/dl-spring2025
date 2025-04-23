import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from peft import PeftModel, PeftConfig
from train import tokenize_function

def inference(model_path, test_data_path):
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = RobertaForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=4)
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = RobertaTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    test_data = test_data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    predictions = classifier(test_data["text"], truncation=True, max_length=256)

    out = {}
    out["ID"] = []
    out["Label"] = []
    for idx, pred in enumerate(predictions.predictions):
        val = pred.argmax(-1)
        out["Label"].append(val)
        out["ID"].append(idx)

    import pandas as pd
    df = pd.DataFrame(out)
    df.to_csv("Results/submission.csv", index=False)