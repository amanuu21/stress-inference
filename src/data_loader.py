# src/data_loader.py
import os
import pickle
import numpy as np
from src.config import DATA_RAW, SUBJECTS, FS

def load_subject(subj_id):
    """
    Load one subject's data from the WESAD dataset.
    Returns: ECG (1D array), EDA (1D), ACC (3 x n_samples), labels (1D)
    """
    pkl_path = os.path.join(DATA_RAW, subj_id, f"{subj_id}.pkl")
    if not os.path.exists(pkl_path):
        return None   # File not found

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Extract signals
    ecg = data['signal']['chest']['ECG'].flatten()
    eda = data['signal']['chest']['EDA'].flatten()
    acc = data['signal']['wrist']['ACC']          # shape (3, n_samples)
    label = data['label'].flatten()

    # Keep only baseline (0) and stress (1) – ignore amusement (2)
    mask = (label == 0) | (label == 1)
    return ecg[mask], eda[mask], acc[:, mask], label[mask]

def generate_synthetic_subject(n_samples=20000):
    """
    Create fake data when real WESAD is not available.
    This lets you test the pipeline without the dataset.
    """
    t = np.linspace(0, n_samples/FS, n_samples)
    # Fake ECG: a sine wave plus noise
    ecg = np.sin(2*np.pi*1.2*t) + 0.5*np.random.randn(n_samples)
    # Fake EDA: slow drift plus noise
    eda = 0.5 + 0.2*np.sin(2*np.pi*0.05*t) + 0.1*np.random.randn(n_samples)
    # Fake ACC: random values
    acc = np.random.randn(3, n_samples) * 0.5
    # Labels: 70% baseline (0), 30% stress (1)
    label = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    return ecg, eda, acc, label

def load_all_subjects():
    """
    Load all subjects from the WESAD dataset.
    If a subject is missing, generate synthetic data for that subject.
    Returns:
        ecg_list, eda_list, acc_list, y_list  – each is a list of arrays per subject
        groups – list of subject IDs repeated for each sample (for cross‑validation)
    """
    ecg_list, eda_list, acc_list, y_list, groups = [], [], [], [], []
    for subj in SUBJECTS:
        data = load_subject(subj)
        if data is None:
            print(f"Subject {subj} not found – generating synthetic data.")
            ecg, eda, acc, y = generate_synthetic_subject()
        else:
            ecg, eda, acc, y = data
        # Store signals (acc is transposed to shape (n_samples, 3))
        ecg_list.append(ecg)
        eda_list.append(eda)
        acc_list.append(acc.T)   # now each row is one sample, 3 columns
        y_list.append(y)
        groups.extend([subj] * len(y))
    return ecg_list, eda_list, acc_list, y_list, groups