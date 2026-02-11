import torch
from torchvision.datasets import PCAM

def getPcamData():
    X_train = PCAM(
    root=r"Pcam",
    split="train",
    download=True
    )

    X_test = PCAM(
        root=r"Pcam",
        download=True
    )

    y_train = X_train.targets
    y_test = X_test.targets

    print(len(X_train))

    return X_train.data, y_train, X_test.data, y_test

getPcamData()