import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def MLPmodel(X_train, y_train, X_test, y_test, batchCount, epochCount):
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1],32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(32,16),
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(16,1)
    )

    optimizer = torch.optim.Adam(model.parameters())#lr = 0.001
    lossFn = torch.nn.BCEWithLogitsLoss()

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.astype(float).values, dtype=torch.float32).reshape(-1, 1)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.astype(float).values, dtype=torch.float32).reshape(-1, 1)

    # Training loop
    epochs = epochCount
    batch_size = batchCount

    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(epochs):
        model.train()
        
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
        
        # Print progress
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