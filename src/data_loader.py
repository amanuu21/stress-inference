# src/data_loader.py
import os
import pickle
import numpy as np
from src.config import DATA_RAW, SUBJECTS, FS

def load_subject(subj_id):
    """Load one subject's data from WESAD."""
    pkl_path = os.path.join(DATA_RAW, subj_id, f"{subj_id}.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    ecg = data['signal']['chest']['ECG'].flatten()
    eda = data['signal']['chest']['EDA'].flatten()
    acc = data['signal']['wrist']['ACC']
    label = data['label'].flatten()
    mask = (label == 0) | (label == 1)
    return ecg[mask], eda[mask], acc[:, mask], label[mask]

def generate_synthetic_subject(n_samples=30000):
    """
    Generate synthetic data with KNOWN features.
    We'll generate the features directly instead of extracting them.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate labels: 70% baseline (0), 30% stress (1)
    # We'll create blocks of stress and baseline
    labels = np.zeros(n_samples, dtype=int)
    # Create 3 stress blocks of 3000 samples each
    for start in [5000, 12000, 20000]:
        end = min(start + 3000, n_samples)
        labels[start:end] = 1
    
    # ----- ECG: artificial signal with clear peaks -----
    t = np.linspace(0, n_samples/FS, n_samples)
    ecg = np.zeros(n_samples)
    
    # Place heartbeats at regular intervals with slight randomness
    beat_positions = []
    pos = 500  # Start after 500 samples
    while pos < n_samples:
        # 0.6 to 1.0 seconds between beats (60-100 BPM)
        interval = int((0.6 + 0.4 * np.random.random()) * FS)
        pos += interval
        if pos < n_samples:
            beat_positions.append(pos)
            # Add a spike
            width = int(0.02 * FS)
            for i in range(-width, width):
                idx = pos + i
                if 0 <= idx < n_samples:
                    ecg[idx] += 1.5 * np.exp(-((i/width)**2) * 8)
    
    # Add baseline wander and noise
    ecg += 0.3 * np.sin(2 * np.pi * 0.15 * t)
    ecg += 0.15 * np.random.randn(n_samples)
    
    # ----- EDA: baseline with stress spikes -----
    eda = 0.5 + 0.05 * np.sin(2 * np.pi * 0.01 * t)
    # Add SCRs (stress responses)
    for pos in np.random.choice(n_samples, size=20, replace=False):
        width = int(0.15 * FS)
        for i in range(-width, width):
            idx = pos + i
            if 0 <= idx < n_samples:
                eda[idx] += 0.2 * np.exp(-((i/width)**2) * 3)
    eda += 0.02 * np.random.randn(n_samples)
    
    # ----- ACC: random movement -----
    acc = np.random.randn(3, n_samples) * 0.3
    
    # ----- Store beat positions for feature calculation -----
    global _SYNTHETIC_BEATS
    _SYNTHETIC_BEATS = np.array(beat_positions)
    
    return ecg, eda, acc, labels

# Global variable for beat positions
_SYNTHETIC_BEATS = None

def get_synthetic_beats():
    return _SYNTHETIC_BEATS

def load_all_subjects():
    """Load all subjects, generating synthetic if missing."""
    ecg_list, eda_list, acc_list, y_list, groups = [], [], [], [], []
    for subj in SUBJECTS:
        data = load_subject(subj)
        if data is None:
            print(f"Subject {subj} not found – generating synthetic data.")
            ecg, eda, acc, y = generate_synthetic_subject()
        else:
            ecg, eda, acc, y = data
        ecg_list.append(ecg)
        eda_list.append(eda)
        acc_list.append(acc.T)
        y_list.append(y)
        groups.extend([subj] * len(y))
    return ecg_list, eda_list, acc_list, y_list, groups