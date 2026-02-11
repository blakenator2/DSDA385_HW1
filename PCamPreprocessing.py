import torch
from torch.utils.data import DataLoader
from torchvision.datasets import PCAM
from torchvision import transforms

def getPcamData():

    transform = transforms.Compose([transforms.ToTensor(), ])
        
    X_train = PCAM(
    root=r"Pcam",
    split="train",
    download=True,
    transform=transform
    )

    X_test = PCAM(
        root=r"Pcam",
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        X_train,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        X_test,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: 96x96x3")
    print(f"Number of classes: 2 (binary classification)")

    return train_loader, test_loader
