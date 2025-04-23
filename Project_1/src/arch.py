import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_conn = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip_conn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip_conn(x)
        out = F.relu(out)
        return out
    
class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout=0.2):
        super(WideResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._create_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_layers(block, 256, num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(256, num_classes)
    
    def _create_layers(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
    
if __name__ == "__main__":
    model = WideResNet(ResidualBlock, [4, 4, 3], num_classes=10, dropout=0.2)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())
    print(f"Number of parameters: {num_params}")

