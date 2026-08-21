# src/baseline.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from src.preprocess import create_dataset
from src.config import REPORTS_DIR
import os

def evaluate_baselines():
    """
    Train three baseline models using Leave-One-Subject-Out cross-validation.
    Prints average F1 and AUROC for each model, and saves a comparison plot.
    """
    print("\n" + "="*60)
    print("STEP 3: BASELINE MODELS")
    print("="*60)
    
    # 1. Load the dataset
    print("Loading dataset...")
    X_ecg, X_eda, X_acc, y, groups = create_dataset()
    
    # 2. Concatenate all features into one big matrix for sklearn
    #    Shape: (total_windows, ECG_DIM + EDA_DIM + ACC_DIM)
    X = np.concatenate([X_ecg, X_eda, X_acc], axis=1)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of unique subjects: {len(np.unique(groups))}")
    
    # 3. Set up Leave-One-Group-Out cross-validation
    #    Each group is one subject (all windows from that subject)
    logo = LeaveOneGroupOut()
    
    # 4. Dictionaries to store results
    results = {
        'Logistic Regression': {'f1': [], 'auc': []},
        'Random Forest': {'f1': [], 'auc': []},
        'SVM': {'f1': [], 'auc': []}
    }
    
    fold = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        fold += 1
        print(f"\nFold {fold}/{len(np.unique(groups))} – testing on subject {groups[test_idx[0]]}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- Model 1: Logistic Regression ---
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        results['Logistic Regression']['f1'].append(f1_score(y_test, y_pred_lr))
        results['Logistic Regression']['auc'].append(roc_auc_score(y_test, y_prob_lr))
        
        # --- Model 2: Random Forest ---
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        results['Random Forest']['f1'].append(f1_score(y_test, y_pred_rf))
        results['Random Forest']['auc'].append(roc_auc_score(y_test, y_prob_rf))
        
        # --- Model 3: SVM ---
        svm = SVC(probability=True, class_weight='balanced', random_state=42)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        y_prob_svm = svm.predict_proba(X_test)[:, 1]
        results['SVM']['f1'].append(f1_score(y_test, y_pred_svm))
        results['SVM']['auc'].append(roc_auc_score(y_test, y_prob_svm))
    
    # 5. Print summary results
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE SUMMARY (LOSO Cross-Validation)")
    print("="*60)
    print(f"{'Model':<22} {'F1 (mean±std)':<18} {'AUROC (mean±std)'}")
    print("-"*60)
    
    for name in results:
        f1_mean = np.mean(results[name]['f1'])
        f1_std = np.std(results[name]['f1'])
        auc_mean = np.mean(results[name]['auc'])
        auc_std = np.std(results[name]['auc'])
        print(f"{name:<22} {f1_mean:.3f} ± {f1_std:.3f}    {auc_mean:.3f} ± {auc_std:.3f}")
    
    # 6. Plot comparison
    plot_baseline_comparison(results)
    
    return results

def plot_baseline_comparison(results):
    """Create a bar chart comparing F1 scores of all baseline models."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    model_names = list(results.keys())
    f1_means = [np.mean(results[m]['f1']) for m in model_names]
    f1_stds = [np.std(results[m]['f1']) for m in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, f1_means, yerr=f1_stds, capsize=10, 
                   color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, f1_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Baseline Model Comparison (Leave-One-Subject-Out)', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save the figure
    save_path = os.path.join(REPORTS_DIR, 'baseline_comparison.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_baselines()