from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import create_data
from arch import WideResNet, ResidualBlock
from utils import mixup_data, mixup_criterion, CESmoothingLoss, augment_train_set

## Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Parameters 

num_epochs = 500
lr = 0.1
weight_decay = 5e-4
num_classes = 10
mixup_alpha = 0.2
label_smoothing = 0.07
train_batch_size = 128
valid_batch_size = 100

## Load the data

train_dataset, valid_dataset = create_data()
train_dataset = augment_train_set(train_dataset, valid_dataset)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

## Load the model

model = WideResNet(ResidualBlock, [4, 4, 3], num_classes=num_classes, dropout=0.2)
model = model.to(device)

## Loss and Optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
loss_fn = CESmoothingLoss(smoothing=label_smoothing)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

## Training Loop

best_val_acc = 0.0
best_model_weights = None
training_history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if mixup_alpha > 0:
            data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha)

        optimizer.zero_grad()
        outputs = model(data)
        if mixup_alpha > 0:
            loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
        else:
            loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    correct = 0
    val_loss = 0.0
    total = 0

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, target)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_acc = 100.0 * correct / total
    training_history["train_loss"].append(avg_train_loss)
    training_history["val_loss"].append(val_loss)
    training_history["val_acc"].append(val_acc)

    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, "best_model_weights.pth")
        print(f"Best model weights saved at epoch {epoch+1}")

## Plot The Training History - 3 Different Plots

basedir = "plots/"
run_name = "test_wide_resnet"

plt.figure(figsize=(10, 7))
plt.plot(training_history["train_loss"], label="Train Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss vs Epochs")
plt.savefig(f"{basedir}/{run_name}_train_loss.png")

plt.figure(figsize=(10, 7))
plt.plot(training_history["val_loss"], label="Val Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Val Loss vs Epochs")
plt.savefig(f"{basedir}/{run_name}_val_loss.png")

plt.figure(figsize=(10, 7))
plt.plot(training_history["val_acc"], label="Val Acc")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Val Accuracy vs Epochs")
plt.savefig(f"{basedir}/{run_name}_val_acc.png")

print("Training Completed!")