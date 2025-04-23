from datasets import load_dataset, concatenate_datasets
from augment import augment_text
import random


def get_augmented_agnews_data(num_samples):
    agnews = load_dataset("ag_news")
    class_names = agnews["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    test_data = agnews["test"]
    train_data = agnews["train"]

    labels = sorted(set(train_data["label"]))
    per_label = num_samples // len(labels)
    indices_by_label = {lab: [] for lab in labels}

    for i, lab in enumerate(train_data["label"]):
        indices_by_label[lab].append(i)
    
    sampled_indices = []
    for lab in labels:
        sampled_indices += random.sample(indices_by_label[lab], per_label)
    
    subset = train_data.select(sampled_indices)

    def perturb(ex):
        return {"text": augment_text(ex["text"])}

    perturbed_subset = subset.map(perturb)

    train_dataset = concatenate_datasets([train_data, perturbed_subset])

    return train_dataset, test_data, id2label, label2id, class_names

