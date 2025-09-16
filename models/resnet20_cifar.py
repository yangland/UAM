# models/resnet20_cifar.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet20_CIFAR"]

def _norm_layer(kind: str, num_channels: int):
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    elif kind == "gn":
        # 4 groups works well for small widths; feel free to tune (e.g., 8)
        groups = min(4, num_channels)
        groups = 1 if num_channels < 4 else groups
        return nn.GroupNorm(groups, num_channels)
    else:
        raise ValueError(f"Unknown norm kind: {kind}")

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, norm="bn"):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.norm1 = _norm_layer(norm, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.norm2 = _norm_layer(norm, out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _norm_layer(norm, out_ch),
            )

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        y = y + self.shortcut(x)
        return F.relu(y, inplace=True)

class ResNet_CIFAR(nn.Module):
    """CIFAR-style ResNet: initial 3x3 conv (stride=1), no maxpool."""
    def __init__(self, block, layers, num_classes=10, norm="bn"):
        super().__init__()
        self.name = f"ResNet{sum(layers)*2+2}_CIFAR_{norm.upper()}"
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = _norm_layer(norm, 16)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, norm=norm)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, out_ch, blocks, stride, norm):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_ch, out_ch, s, norm=norm))
            self.in_ch = out_ch * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

def ResNet20_CIFAR(num_classes=10, norm="gn"):
    """norm: 'bn' (standard) or 'gn' (non-IID friendly)."""
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes, norm=norm)

if __name__ == "__main__":
    for kind in ["bn", "gn"]:
        model = ResNet20_CIFAR(num_classes=10, norm=kind)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        nparams = sum(p.numel() for p in model.parameters())
        print(model.name, "| params:", nparams, "| out:", y.shape)

    print("=== module names ===")
    for name, _ in model.named_modules():
        print(name)

    print("\n=== state_dict keys (weights/buffers) ===")
    for k in model.state_dict().keys():
        print(k)
