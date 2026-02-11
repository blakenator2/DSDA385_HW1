import pandas as pd
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def PcamCNNModel(X_train, y_train, X_test, y_test, epochCount, learn):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),  
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), 
        torch.nn.Dropout2d(0.2),
        
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), 
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), 
        torch.nn.Dropout2d(0.2),
    
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), 
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), 
        torch.nn.Dropout2d(0.2),
        
        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),  
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), 
        torch.nn.Dropout2d(0.2),
        
        torch.nn.Flatten(), 
        
        torch.nn.Linear(9216, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        
        torch.nn.Linear(512, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        
        torch.nn.Linear(128, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = learn)
    lossFn = torch.nn.BCEWithLogitsLoss()
    
    X_train_tensor = torch.tensor(X_train).float() /255.0
    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(1)

    X_test_tensor = torch.tensor(X_test).float() / 255.0
    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(y_test).float().unsqueeze(1)

    print("Pre-processing done")

    # Training loop
    epochs = epochCount
    batch_size = 64

    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(epochs):
        model.train()
        
        running_train_loss = 0.0
        num_batches = 0

        # --- Training ---
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            outputs = model(X_batch)
            loss = lossFn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_batches += 1

        epoch_train_loss = running_train_loss / num_batches
        train_losses.append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                X_val_batch = X_test_tensor[i:i+batch_size]
                y_val_batch = y_test_tensor[i:i+batch_size]

                val_outputs = model(X_val_batch)
                val_loss = lossFn(val_outputs, y_val_batch)

                running_val_loss += val_loss.item()
                val_batches += 1

        epoch_val_loss = running_val_loss / val_batches
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] '
                f'Train Loss: {epoch_train_loss:.4f} '
                f'Val Loss: {epoch_val_loss:.4f}')