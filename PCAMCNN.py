import pandas as pd
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def PcamCNNModel(X_train, y_train, X_test, y_test, epochCount, learn):
    model = torch.nn.Sequential(
        # Input: (batch, 3, 96, 96)
        
        # Conv Block 1
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),  # -> (32, 96, 96)
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # -> (32, 48, 48)
        torch.nn.Dropout2d(0.2),
        
        # Conv Block 2
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 48, 48)
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # -> (64, 24, 24)
        torch.nn.Dropout2d(0.2),
        
        # Conv Block 3
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (128, 24, 24)
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # -> (128, 12, 12)
        torch.nn.Dropout2d(0.3),
        
        # Conv Block 4
        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> (256, 12, 12)
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # -> (256, 6, 6)
        torch.nn.Dropout2d(0.3),
        
        # Flatten and Dense layers
        torch.nn.Flatten(),  # -> (256 * 6 * 6 = 9216)
        
        torch.nn.Linear(9216, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        
        torch.nn.Linear(512, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        
        torch.nn.Linear(128, 1)  # Binary classification
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = learn)
    lossFn = torch.nn.BCEWithLogitsLoss()
    
    X_train_tensor = torch.tensor(X_train).float() /255.0
    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train).long()

    X_test_tensor = torch.tensor(X_test).float() / 255.0
    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(y_test).long()

    print("Pre-processing done")

    # Training loop
    epochs = epochCount
    batch_size = 64

    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(epochs):
        model.train()
        print(epoch)

        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            # Get batch
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(X_batch)
            loss = lossFn(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = lossFn(model(X_train_tensor), y_train_tensor)
            val_loss = lossFn(model(X_test_tensor), y_test_tensor)
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            train_outputs = model(X_train_tensor)
            train_loss = lossFn(train_outputs, y_train_tensor)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.4f}')

    print("Total time to train in seconds: ", time.time()-start)

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(X_test_tensor))
        predicted_classes = (predictions > 0.5).float()
        
        # Convert to numpy arrays for sklearn
        y_pred = predicted_classes.cpu().numpy().flatten()  # or .squeeze()
        y_true = y_test_tensor.cpu().numpy().flatten()
        
        # Sklearn metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Classification Report
        print('\nClassification Report:')
        print(classification_report(y_true, y_pred))
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')