# src/ablation.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, roc_auc_score
from src.preprocess import create_dataset
from src.model import MultimodalFusionModel
from src.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, REPORTS_DIR
import os
import matplotlib.pyplot as plt

def train_ablation_fold(X_ecg, X_eda, X_acc, y, train_idx, val_idx, target_modality):
    """
    Train the multimodal model but only use ONE modality.
    The other two modalities are replaced with zeros.
    
    target_modality: 'ecg', 'eda', or 'acc'
    """
    # Convert to tensors
    X_ecg_t = torch.tensor(X_ecg[train_idx], dtype=torch.float32)
    X_eda_t = torch.tensor(X_eda[train_idx], dtype=torch.float32)
    X_acc_t = torch.tensor(X_acc[train_idx], dtype=torch.float32)
    y_t = torch.tensor(y[train_idx], dtype=torch.long)
    
    dataset = TensorDataset(X_ecg_t, X_eda_t, X_acc_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultimodalFusionModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        for batch in loader:
            ecg_b, eda_b, acc_b, y_b = batch
            
            # ---- ABLATION: zero out the non‑target modalities ----
            if target_modality == 'ecg':
                # Keep ECG, zero out EDA and ACC
                eda_b = torch.zeros_like(eda_b)
                acc_b = torch.zeros_like(acc_b)
            elif target_modality == 'eda':
                # Keep EDA, zero out ECG and ACC
                ecg_b = torch.zeros_like(ecg_b)
                acc_b = torch.zeros_like(acc_b)
            elif target_modality == 'acc':
                # Keep ACC, zero out ECG and EDA
                ecg_b = torch.zeros_like(ecg_b)
                eda_b = torch.zeros_like(eda_b)
            
            optimizer.zero_grad()
            logits = model(ecg_b, eda_b, acc_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_ecg_val = torch.tensor(X_ecg[val_idx], dtype=torch.float32)
        X_eda_val = torch.tensor(X_eda[val_idx], dtype=torch.float32)
        X_acc_val = torch.tensor(X_acc[val_idx], dtype=torch.float32)
        
        # Zero out non‑target modalities for validation too
        if target_modality == 'ecg':
            X_eda_val = torch.zeros_like(X_eda_val)
            X_acc_val = torch.zeros_like(X_acc_val)
        elif target_modality == 'eda':
            X_ecg_val = torch.zeros_like(X_ecg_val)
            X_acc_val = torch.zeros_like(X_acc_val)
        elif target_modality == 'acc':
            X_ecg_val = torch.zeros_like(X_ecg_val)
            X_eda_val = torch.zeros_like(X_eda_val)
        
        logits = model(X_ecg_val, X_eda_val, X_acc_val)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=1).numpy()
    
    f1 = f1_score(y[val_idx], preds)
    auc = roc_auc_score(y[val_idx], probs)
    return f1, auc

def run_ablation():
    """
    Run ablation study: test each modality alone.
    Compares ECG‑only, EDA‑only, ACC‑only.
    """
    print("\n" + "="*60)
    print("STEP 5: ABLATION STUDY")
    print("="*60)
    print("Testing each modality alone to see how important it is.")
    
    # Load dataset
    print("Loading dataset...")
    X_ecg, X_eda, X_acc, y, groups = create_dataset()
    print(f"Total windows: {len(X_ecg)}")
    
    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)
    
    modalities = ['ecg', 'eda', 'acc']
    modality_names = {'ecg': 'ECG only', 'eda': 'EDA only', 'acc': 'ACC only'}
    results = {m: {'f1': [], 'auc': []} for m in modalities}
    
    for target in modalities:
        print(f"\n--- Ablation: {modality_names[target]} ---")
        fold = 0
        for train_idx, val_idx in logo.split(X_ecg, y, groups):
            fold += 1
            f1, auc = train_ablation_fold(X_ecg, X_eda, X_acc, y, train_idx, val_idx, target)
            results[target]['f1'].append(f1)
            results[target]['auc'].append(auc)
            
            if fold % 5 == 0:  # Print progress every 5 folds
                print(f"  Fold {fold}/{len(unique_subjects)} – F1: {f1:.3f}, AUC: {auc:.3f}")
        
        avg_f1 = np.mean(results[target]['f1'])
        avg_auc = np.mean(results[target]['auc'])
        print(f"  Average F1: {avg_f1:.3f}  AUROC: {avg_auc:.3f}")
    
    # ---- Print summary table ----
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'Modality':<12} {'F1 (mean±std)':<18} {'AUROC (mean±std)'}")
    print("-"*60)
    for target in modalities:
        f1_mean = np.mean(results[target]['f1'])
        f1_std = np.std(results[target]['f1'])
        auc_mean = np.mean(results[target]['auc'])
        auc_std = np.std(results[target]['auc'])
        print(f"{modality_names[target]:<12} {f1_mean:.3f} ± {f1_std:.3f}    {auc_mean:.3f} ± {auc_std:.3f}")
    print("="*60)
    
    # ---- Plot the results ----
    plot_ablation_results(results)
    
    return results

def plot_ablation_results(results):
    """Bar chart comparing all single‑modality performances."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Full model result from Step 4 – we'll hardcode a placeholder since we don't have it saved
    # In a real project, you'd load it from a file. For now, we'll just plot the ablation.
    modality_names = ['ECG only', 'EDA only', 'ACC only']
    f1_means = [np.mean(results['ecg']['f1']), np.mean(results['eda']['f1']), np.mean(results['acc']['f1'])]
    f1_stds = [np.std(results['ecg']['f1']), np.std(results['eda']['f1']), np.std(results['acc']['f1'])]
    
    # Add the full model result from Step 4 (we'll approximate it as slightly higher)
    # In reality, you'd load this from a saved log. For now, we'll add a placeholder.
    # But to be honest, we'll just print a note.
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modality_names, f1_means, yerr=f1_stds, capsize=10, 
                   color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    
    for bar, val in zip(bars, f1_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Ablation Study: Single Modality Performance', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add a note about the full model
    plt.figtext(0.5, -0.05, "Note: These are single‑modality results. Compare with the full multimodal model from Step 4.",
                ha='center', fontsize=10, style='italic')
    
    save_path = os.path.join(REPORTS_DIR, 'ablation_results.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_ablation()