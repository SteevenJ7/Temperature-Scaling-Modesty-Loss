from torchvision import transforms, datasets
import torch


def load_data(d, train=False, batch_size=100):
    """ Create and return dataloader for different dataset """
    if d == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        data_set = datasets.CIFAR10(root='Dataset/', train=train, download=True, transform=transform)
        return torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
    elif d == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        data_set = datasets.CIFAR100(root='Dataset/', train=train, download=True, transform=transform)
        return torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
    elif d == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return torch.utils.data.DataLoader(datasets.ImageFolder("Dataset/ILSVRC", transform), batch_size=batch_size,
                                           shuffle=False)
    elif d == "SVHN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if not train:
            data_set = datasets.SVHN(root='Dataset/', split="test", download=True, transform=transform)
        else:
            data_set = datasets.SVHN(root='Dataset/', split="train", download=True, transform=transform)
        return torch.utils.data.DataLoader(data_set, batch_size=batch_size)
    else:
        raise TypeError("Dataset inconnu")
