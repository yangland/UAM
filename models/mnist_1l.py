import torch.nn as nn
import torch.nn.functional as F
class Mnist_1L(nn.Module):
    def __init__(self):
        self.name = "Mnist 1 layers"
        super(Mnist_1L, self).__init__()
        self.fc1 = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    model=Mnist_1L()
    print(model)
    print(model.name)