import h5py
import numpy as np
from pathlib import Path

def getPcamData():
    download_dir = Path(r"Pcam")
    
    with h5py.File(download_dir / "camelyonpatch_level_2_split_train_x.h5", 'r') as f:
        X_train = f['x'][:]
    
    with h5py.File(download_dir / "camelyonpatch_level_2_split_train_y.h5", 'r') as f:
        y_train = f['y'][:].squeeze() 

    with h5py.File(download_dir / "camelyonpatch_level_2_split_test_x.h5", 'r') as f:
        X_test = f['x'][:]
    
    with h5py.File(download_dir / "camelyonpatch_level_2_split_test_y.h5", 'r') as f:
        y_test = f['y'][:].squeeze()
    
    print(f"\nData loaded successfully!")
    print(f"X_train shape: {X_train.shape}")  # Should be (N, 96, 96, 3)
    print(f"y_train shape: {y_train.shape}")  # Should be (N,)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Label distribution - Train: {np.bincount(y_train)}")
    print(f"Label distribution - Test: {np.bincount(y_test)}")

    train_half = len(X_train) // 2
    test_half = len(X_test) // 2

    X_train = X_train[:train_half]
    y_train = y_train[:train_half]

    X_test = X_test[:test_half]
    y_test = y_test[:test_half]
    
    return X_train, y_train, X_test, y_test

