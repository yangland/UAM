# models/mobilenetv2_cifar.py
import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import mobilenet_v2

class MobileNetV2_CIFAR(nn.Module):
    """
    CIFAR-10 friendly MobileNetV2:
    - width_mult controls model size (e.g., 0.5)
    - stem conv stride=1 (32x32)
    - classifier -> logits for num_classes
    """
    def __init__(self, num_classes=10, in_ch=3, width_mult=0.5, dropout=0.2):
        super().__init__()
        self.name = f"MobileNetV2_CIFAR_w{width_mult}"
        m = mobilenet_v2(weights=None, width_mult=width_mult)
        # Stem: 3x3 conv, change in_channels and stride for CIFAR
        # torchvision layout: features[0][0] is the first Conv2d
        first_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(in_ch, first_conv.out_channels,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        # Classifier: Dropout + Linear to num_classes
        in_feat = m.classifier[1].in_features  # usually 1280
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, num_classes, bias=True),
        )
        # Kaiming for new head
        nn.init.kaiming_normal_(m.classifier[1].weight, nonlinearity="relu")
        nn.init.zeros_(m.classifier[1].bias)
        self.features = m.features
        self.classifier = m.classifier

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = MobileNetV2_CIFAR(num_classes=10, in_ch=3, width_mult=0.5, dropout=0.2)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("logits:", y.shape)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    
    print("=== module names ===")
    for name, _ in model.named_modules():
        print(name)

    print("\n=== state_dict keys (weights/buffers) ===")
    for k in model.state_dict().keys():
        print(k)
