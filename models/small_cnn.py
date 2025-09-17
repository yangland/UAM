import torch
import torch.nn as nn
import torch.nn.functional as F

def norm_layer(c, kind="group", groups=8):
    if kind == "group":
        g = min(groups, c) if c % groups != 0 else groups
        return nn.GroupNorm(num_groups=g, num_channels=c, affine=True)
    elif kind == "batch":
        return nn.BatchNorm2d(c)
    elif kind == "layer":
        return nn.GroupNorm(num_groups=1, num_channels=c, affine=True)
    else:
        return nn.Identity()

class SmallCNN_MNIST(nn.Module):
    """
    Input: (N, 1, 28, 28)
    Params: ~95k
    """
    def __init__(self, num_classes=10, norm="group", dropout_p=0.1):
        super().__init__()
        self.name = "SmallCNN_MNIST"

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, norm),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 28x28 -> 14x14
            nn.Dropout(dropout_p),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 14x14 -> 7x7
            nn.Dropout(dropout_p),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> (N,128,1,1)
        )
        self.fc = nn.Linear(128, num_classes)

    def _ensure_nchw(self, x):
        # Accepts [N,784], [N,28,28], or [N,1,28,28] and returns [N,1,28,28]
        if x.dim() == 2 and x.size(1) == 28*28:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3 and x.size(1) == 28 and x.size(2) == 28:
            x = x.unsqueeze(1)
        return x

    def forward(self, x):
        x = self._ensure_nchw(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x).flatten(1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model=SmallCNN_MNIST()
    print(model)
    print(model.name)
    
    print("=== module names ===")
    for name, _ in model.named_modules():
        print(name)

    print("\n=== state_dict keys (weights/buffers) ===")
    for k in model.state_dict().keys():
        print(k)
