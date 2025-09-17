import torch
import torch.nn as nn
import torch.nn.functional as F

def norm2d(c, kind="group", groups=4):
    if kind == "group":
        g = min(groups, c)
        while c % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, c, affine=True)
    elif kind == "layer":
        return nn.GroupNorm(1, c, affine=True)
    else:
        return nn.Identity()

class TinyCNN_MNIST(nn.Module):
    """
    Ultra-small CNN for MNIST
    - No BatchNorm (avoids cross-client stat drift)
    - GlobalAvgPool head keeps FC tiny
    - Handles [N,784], [N,28,28], or [N,1,28,28] inputs

    ~5.1K params @ width_mult=1.0 (C=16→32)
    """
    def __init__(self, num_classes=10, norm="layer", dropout_p=0.05, width_mult=1.0):
        super().__init__()
        self.name = "TinyCNN_MNIST"

        c1 = int(16 * width_mult)
        c2 = int(32 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1, bias=False),   # 1→c1
            norm2d(c1, norm),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),  # c1→c1
            norm2d(c1, norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28→14
            nn.Dropout(dropout_p),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),  # c1→c2
            norm2d(c2, norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14→7
            nn.Dropout(dropout_p),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(c2, num_classes)

    @staticmethod
    def _ensure_nchw(x):
        if x.dim() == 2 and x.size(1) == 28*28:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3 and x.shape[1:] == (28, 28):
            x = x.unsqueeze(1)
        return x

    def forward(self, x):
        x = self._ensure_nchw(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = TinyCNN_MNIST()
    print(model)
    print("params:", count_params(model))

    print("=== module names ===")
    for name, _ in model.named_modules():
        print(name)

    print("\n=== state_dict keys (weights/buffers) ===")
    for k in model.state_dict().keys():
        print(k)
