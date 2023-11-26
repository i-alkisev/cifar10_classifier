import torchvision

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]
)

dataset_cifar10_train = torchvision.datasets.CIFAR10(
    root="./data/", train=True, transform=transform, download=False
)
dataset_cifar10_test = torchvision.datasets.CIFAR10(
    root="./data/", train=False, transform=transform, download=False
)
