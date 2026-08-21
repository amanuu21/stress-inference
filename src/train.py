# src/train.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, roc_auc_score
from src.preprocess import create_dataset
from src.model import MultimodalFusionModel
from src.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODELS_DIR
import os

def train_one_fold(X_ecg, X_eda, X_acc, y, train_idx, val_idx):
    """
    Train the multimodal neural network on one fold.
    Returns: F1 score, AUROC, and the trained model.
    """
    # Convert to PyTorch tensors
    X_ecg_t = torch.tensor(X_ecg[train_idx], dtype=torch.float32)
    X_eda_t = torch.tensor(X_eda[train_idx], dtype=torch.float32)
    X_acc_t = torch.tensor(X_acc[train_idx], dtype=torch.float32)
    y_t = torch.tensor(y[train_idx], dtype=torch.long)
    
    # Create DataLoader
    dataset = TensorDataset(X_ecg_t, X_eda_t, X_acc_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model, optimizer, loss function
    model = MultimodalFusionModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            ecg_b, eda_b, acc_b, y_b = batch
            optimizer.zero_grad()
            logits = model(ecg_b, eda_b, acc_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} – Loss: {epoch_loss/len(loader):.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_ecg_val = torch.tensor(X_ecg[val_idx], dtype=torch.float32)
        X_eda_val = torch.tensor(X_eda[val_idx], dtype=torch.float32)
        X_acc_val = torch.tensor(X_acc[val_idx], dtype=torch.float32)
        logits = model(X_ecg_val, X_eda_val, X_acc_val)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=1).numpy()
    
    f1 = f1_score(y[val_idx], preds)
    auc = roc_auc_score(y[val_idx], probs)
    return f1, auc, model

def run_cross_validation():
    """
    Run Leave-One-Subject-Out cross-validation for the neural network.
    Saves the best model and prints average performance.
    """
    print("\n" + "="*60)
    print("STEP 4: MULTIMODAL NEURAL NETWORK")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    X_ecg, X_eda, X_acc, y, groups = create_dataset()
    print(f"Total windows: {len(X_ecg)}")
    
    # Set up LOSO cross-validation
    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)
    
    f1_scores = []
    auc_scores = []
    best_model = None
    best_f1 = 0.0
    
    fold = 0
    for train_idx, val_idx in logo.split(X_ecg, y, groups):
        fold += 1
        print(f"\nFold {fold}/{len(unique_subjects)} – testing on subject {groups[val_idx[0]]}")
        
        f1, auc, model = train_one_fold(X_ecg, X_eda, X_acc, y, train_idx, val_idx)
        f1_scores.append(f1)
        auc_scores.append(auc)
        print(f"  F1 = {f1:.3f}, AUROC = {auc:.3f}")
        
        # Keep the best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    
    # Summary statistics
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print("\n" + "="*60)
    print("MULTIMODAL NN PERFORMANCE SUMMARY (LOSO)")
    print("="*60)
    print(f"F1:    {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"AUROC: {avg_auc:.3f} ± {std_auc:.3f}")
    print("="*60)
    
    # Save the best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'best_model.pth')
    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to: {model_path}")
    
    return best_model, avg_f1, avg_auc

if __name__ == "__main__":
    run_cross_validation()