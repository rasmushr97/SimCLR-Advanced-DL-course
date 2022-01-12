import torch.nn as nn
from torchvision.models.resnet import resnet50

class Cifar10_SimCLR_Model(nn.Module):
    def __init__(self):
        super(Cifar10_SimCLR_Model, self).__init__()

        enc_layers = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                enc_layers.append(module)

        self.encoder  = nn.Sequential(*enc_layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

