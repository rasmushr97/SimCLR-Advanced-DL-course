from simclr import SimCLR
from data import create_cifar_train_loader
import torch
import torch.nn as nn
from models import Cifar10_SimCLR_model

def main():
    # load data
    dataloader = create_cifar_train_loader()

    # intialize model
    model = Cifar10_SimCLR_Model()

    # train model
    trainer = SimCLR(model)
    trainer.train(dataloader, epochs=1)




if __name__ == '__main__':
    main()