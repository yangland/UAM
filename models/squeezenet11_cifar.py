# models/squeezenet11_cifar.py
import torch
import torch.nn as nn
from torchvision.models.squeezenet import squeezenet1_1

class Head(nn.Module):
    def __init__(self, num_classes=10, in_channels=512, p=0.5):
        super().__init__()
        self.drop = nn.Dropout(p=p)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        # x = self.relu(x)
        x = self.gap(x)
        return torch.flatten(x, 1)  # [N, num_classes]

class SqueezeNet11_CIFAR(nn.Module):
    """
    CIFAR-10 friendly SqueezeNet v1.1:
    - stem conv stride=1
    - remove first MaxPool
    - custom head -> logits
    """
    def __init__(self, num_classes=10, in_ch=3, dropout=0.5):
        super().__init__()
        self.name = "SqueezeNet11_CIFAR"
        m = squeezenet1_1(weights=None)
        # stem conv to match channels and keep stride=1 for 32x32
        m.features[0] = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # remove the FIRST MaxPool2d (v1.1: index 2)
        for i, layer in enumerate(m.features):
            if isinstance(layer, nn.MaxPool2d):
                m.features[i] = nn.Identity()
                break
        self.features = m.features
        self.head = Head(num_classes=num_classes, in_channels=512, p=dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)  # logits
        return x


if __name__ == "__main__":
    model = SqueezeNet11_CIFAR(num_classes=10, in_ch=3, dropout=0.5)

    print("=== module names ===")
    for name, _ in model.named_modules():
        print(name)

    print("\n=== state_dict keys (weights/buffers) ===")
    for k in model.state_dict().keys():
        print(k)
