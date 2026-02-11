from torchvision.datasets import CIFAR100

def getData():
    X_train = CIFAR100(
    root=r"cifar-100",
    train=True,
    transform=None,
    target_transform=None,
    download=True
    )

    X_test = CIFAR100(
        root=r"cifar-100",
        train=False,
        download=True
    )

    y_train = X_train.targets
    y_test = X_test.targets

    return X_train.data, y_train, X_test.data, y_test
