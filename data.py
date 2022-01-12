import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class PairDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        aug1 = self.transform(img)
        aug2 = self.transform(img)

        return aug1, aug2, label


def create_cifar_train_loader():
    cifar10_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        # TODO edit colorJitter
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True
    )

    trainset = PairDataset(trainset, cifar10_train_transform)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    return trainloader


def create_cifar_test_loader():
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=cifar10_test_transform)

    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return testloader