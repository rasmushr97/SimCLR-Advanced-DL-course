from simclr import SimCLR
from data import create_cifar_train_loader

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

def main():

    # load data
    dataloader = create_cifar_train_loader()

    # intialize model
    encoder = resnet18(pretrained=False)
    model = nn.Sequential(
        encoder,
        nn.Linear(1000, 1000, bias=False),
        nn.ReLU(),
        nn.Linear(1000, 128, bias=False),
    )

    # train model
    trainer = SimCLR(model)
    trainer.train(dataloader, epochs=1)





if __name__ == '__main__':
    main()