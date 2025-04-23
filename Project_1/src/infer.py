import pandas as pd
import pickle
import torch
from arch import WideResNet, ResidualBlock
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/data_aug_best_model.pth"

model = WideResNet(ResidualBlock, [4, 4, 3], num_classes=10, dropout=0.2)
model.load_state_dict(torch.load(model_path))

model.to(device)

model.eval()

test_data_path = "data/cifar_test_nolabel.pkl"
test_data = pickle.load(open(test_data_path, "rb"))[b'data']

test_time_transforms = [
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ]),
    ]

out = {}
out["ID"] = []
out["Labels"] = []
for idx, sample in enumerate(test_data):

    with torch.no_grad():
        img = Image.fromarray(sample)
        test_time_transform_outputs = []
        for transform in test_time_transforms:
            input_tensor = transform(img).unsqueeze(0).to(device)
            output = model(input_tensor)
            test_time_transform_outputs.append(output)
        
        avg_output = torch.mean(torch.stack(test_time_transform_outputs), dim=0)
        _, predicted = torch.max(avg_output.data, 1) 
        out["ID"].append(idx)
        out["Labels"].append(predicted.item())

df = pd.DataFrame(out)
df.to_csv("results/submission.csv", index=False)
print("File saved successfully")