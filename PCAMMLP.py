import pandas as pd
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def PcamMLPModel(X_train, y_train, X_test, y_test, epochCount, learn):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(27648, 2048),  # PCAM is 96x96x3 = 27648
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        
        torch.nn.Linear(2048, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        
        torch.nn.Linear(1024, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        
        torch.nn.Linear(256, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = learn)
    lossFn = torch.nn.BCEWithLogitsLoss()
    
    X_train_tensor = torch.tensor(X_train).float() /255.0
    X_train_tensor = X_train_tensor.reshape(X_train_tensor.shape[0], -1)
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(1)


    X_test_tensor = torch.tensor(X_test).float() / 255.0
    X_test_tensor = X_test_tensor.reshape(X_test_tensor.shape[0], -1)
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
        print(epoch)
        
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
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(X_test_tensor))
        predicted_classes = torch.argmax(predictions, dim=1)
        
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
        num_classes = len(np.unique(y_true))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
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

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')