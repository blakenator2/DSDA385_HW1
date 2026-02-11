import pandas as pd
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def PcamMLPModel(train_loader, test_loader, epochCount, learn):
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
        
        torch.nn.Linear(256, 1)  # Binary classification
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learn)
    lossFn = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Starting training...")
    start = time.time()
    
    for epoch in range(epochCount):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            outputs = model(X_batch)
            loss = lossFn(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = lossFn(outputs, y_batch)
                
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_loss = val_loss / len(test_loader)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochCount}], Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    print(f"Total training time: {time.time()-start:.2f}s")