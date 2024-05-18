import torch
import torch.nn as nn
import torch.nn.functional as F


class Kmnist_12L(nn.Module):
    def __init__(self):
        self.name = "Kmnist 12 layers"
        super(Kmnist_12L, self).__init__()
        # 28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # 28
        # 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 14
        # 7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # 5
        # 3 pool2
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        # 3
        self.conv5 = nn.Conv2d(96, 96, kernel_size=2, stride=1, padding=0)
        # 2

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(384, 270)
        self.fc2 = nn.Linear(270, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 49)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

if __name__ == '__main__':
    model = Kmnist_12L()
    print(model)
