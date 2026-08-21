# src/preprocess.py
import numpy as np
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from src.config import FS, WINDOW_SEC, STEP_SEC, ECG_DIM, EDA_DIM, ACC_DIM, MODELS_DIR
from src.data_loader import load_all_subjects
import joblib
import os

def extract_ecg_features(ecg_signal):
    """
    Extract 5 Heart Rate Variability (HRV) features from a 60-second ECG window.
    
    These features measure how the heart rate changes over time.
    Low HRV usually means stress or fatigue.
    """
    try:
        # Step 1: Clean the signal (remove noise)
        cleaned = nk.ecg_clean(ecg_signal, sampling_rate=FS)
        
        # Step 2: Find the "R-peaks" – the main spikes of each heartbeat
        rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=FS)['ECG_R_Peaks']
        
        # If we didn't find at least 2 heartbeats, skip this window
        if len(rpeaks) < 2:
            return np.full(ECG_DIM, np.nan)
        
        # Step 3: Calculate HRV metrics from the time between heartbeats
        hrv = nk.hrv(rpeaks, sampling_rate=FS, show=False)
        
        # Extract these 5 specific features:
        # 1. MeanNN   = Average time between heartbeats (milliseconds)
        # 2. SDNN     = Standard deviation of heartbeats – overall HRV
        # 3. RMSSD    = Short-term HRV – linked to relaxation
        # 4. pNN50    = Percentage of heartbeats with big changes – stress indicator
        # 5. MeanHR   = Average heart rate (beats per minute)
        features = [
            hrv.get('HRV_MeanNN', [np.nan])[0],
            hrv.get('HRV_SDNN', [np.nan])[0],
            hrv.get('HRV_RMSSD', [np.nan])[0],
            hrv.get('HRV_pNN50', [np.nan])[0],
            hrv.get('HRV_MeanHR', [np.nan])[0]
        ]
        return np.array(features)
    except:
        # If anything breaks, return NaN (Not a Number)
        return np.full(ECG_DIM, np.nan)

def extract_eda_features(eda_signal):
    """
    Extract 5 Electrodermal Activity (EDA) features from a 60-second window.
    
    EDA measures skin conductance – when you sweat from stress, this goes up.
    """
    try:
        # Clean the EDA signal
        cleaned = nk.eda_clean(eda_signal, sampling_rate=FS)
        
        # Split EDA into two parts:
        # - Tonic: slow, baseline sweating (like your usual state)
        # - Phasic: fast spikes (reactions to sudden stress)
        phasic, tonic, scr_peaks = nk.eda_phasic(cleaned, sampling_rate=FS)
        
        # Count how many stress spikes (SCRs) happened in this 60 seconds
        scr_count = len(scr_peaks)
    except:
        # If it fails, use raw signal and assume 0 spikes
        cleaned = eda_signal
        tonic = np.full_like(eda_signal, np.nan)
        scr_count = 0
    
    # Extract these 5 features:
    # 1. Mean     = Average skin conductance level
    # 2. Std      = How much it fluctuated
    # 3. Range    = Max - Min (how wide the swings were)
    # 4. Tonic    = Baseline sweating level
    # 5. SCR count = Number of stress spikes
    features = [
        np.mean(cleaned),
        np.std(cleaned),
        np.max(cleaned) - np.min(cleaned),
        np.mean(tonic) if not np.isnan(tonic).all() else np.nan,
        scr_count
    ]
    return np.array(features)

def extract_acc_features(acc_signal):
    """
    Extract 8 movement features from the 3-axis accelerometer.
    
    acc_signal has shape (window_length, 3) – each row is X, Y, Z.
    """
    # Mean movement on each axis (average X, average Y, average Z)
    mean_axes = np.mean(acc_signal, axis=0)
    
    # Standard deviation (how shaky) on each axis
    std_axes = np.std(acc_signal, axis=0)
    
    # Magnitude = total movement intensity (sqrt(X^2 + Y^2 + Z^2))
    magnitude = np.sqrt(np.sum(acc_signal**2, axis=1))
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)
    
    # Return 8 numbers: [mean_x, mean_y, mean_z, std_x, std_y, std_z, mean_mag, std_mag]
    return np.concatenate([mean_axes, std_axes, [mean_mag, std_mag]])

def window_and_extract(ecg, eda, acc, label):
    """
    Slide a 60-second window over the signals, extract features for each window.
    This is the main function that turns raw signals into a feature table.
    """
    window_samples = WINDOW_SEC * FS   # 60 * 700 = 42,000 samples per window
    step_samples = STEP_SEC * FS       # 30 * 700 = 21,000 samples step (50% overlap)
    n = len(ecg)  # Total length of the recording
    
    # Lists to store features from each window
    X_ecg, X_eda, X_acc, y = [], [], [], []
    
    # Slide the window from start to end, jumping by `step_samples`
    for start in range(0, n - window_samples + 1, step_samples):
        end = start + window_samples
        
        # Cut out the 60-second chunk from each signal
        ecg_win = ecg[start:end]
        eda_win = eda[start:end]
        acc_win = acc[start:end, :]
        
        # What is the label for this window?
        # We take the "majority vote" – if most of the 60 seconds is stress, label = 1.
        win_label = np.bincount(label[start:end].astype(int)).argmax()
        
        # Extract features from this window
        ecg_feat = extract_ecg_features(ecg_win)
        eda_feat = extract_eda_features(eda_win)
        acc_feat = extract_acc_features(acc_win)
        
        # If any feature is NaN (e.g., not enough heartbeats), skip this window
        if np.isnan(ecg_feat).any() or np.isnan(eda_feat).any():
            continue
        
        # Store the features
        X_ecg.append(ecg_feat)
        X_eda.append(eda_feat)
        X_acc.append(acc_feat)
        y.append(win_label)
    
    # Convert lists to numpy arrays
    return (np.array(X_ecg), np.array(X_eda), np.array(X_acc), np.array(y))

def create_dataset():
    """
    The master function that:
    1. Loads all subjects (real or synthetic)
    2. Windows each subject into 60-second chunks
    3. Extracts features from each window
    4. Scales the features (makes them all in the same range, e.g., 0 to 1)
    5. Returns the final dataset ready for AI training
    """
    # Load raw signals from the data_loader
    ecg_list, eda_list, acc_list, y_list, groups = load_all_subjects()
    
    all_ecg, all_eda, all_acc, all_y, all_groups = [], [], [], [], []
    
    # Process each subject one by one
    for i, (ecg, eda, acc, y) in enumerate(zip(ecg_list, eda_list, acc_list, y_list)):
        print(f"Processing subject {i+1}/{len(ecg_list)}...")
        X_ecg, X_eda, X_acc, y_sub = window_and_extract(ecg, eda, acc, y)
        
        all_ecg.append(X_ecg)
        all_eda.append(X_eda)
        all_acc.append(X_acc)
        all_y.append(y_sub)
        # groups keeps track of which subject each window came from
        all_groups.extend([str(i)] * len(X_ecg))
    
    # Concatenate all subjects into one giant dataset
    X_ecg = np.vstack(all_ecg)
    X_eda = np.vstack(all_eda)
    X_acc = np.vstack(all_acc)
    y = np.hstack(all_y)
    groups = np.array(all_groups)
    
    # Scale each modality separately
    # Scaling means: subtract the average, divide by the standard deviation.
    # This makes sure ECG features (which might be 0-100) and EDA (0-10) are on the same scale.
    print("Scaling ECG features...")
    scaler_ecg = StandardScaler().fit(X_ecg)
    X_ecg = scaler_ecg.transform(X_ecg)
    
    print("Scaling EDA features...")
    scaler_eda = StandardScaler().fit(X_eda)
    X_eda = scaler_eda.transform(X_eda)
    
    print("Scaling ACC features...")
    scaler_acc = StandardScaler().fit(X_acc)
    X_acc = scaler_acc.transform(X_acc)
    
    # Save the scalers so we can use them later for new data (API/dashboard)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler_ecg, os.path.join(MODELS_DIR, 'scaler_ecg.pkl'))
    joblib.dump(scaler_eda, os.path.join(MODELS_DIR, 'scaler_eda.pkl'))
    joblib.dump(scaler_acc, os.path.join(MODELS_DIR, 'scaler_acc.pkl'))
    print(f"Scalers saved to {MODELS_DIR}/")
    
    print(f"Dataset ready: {len(X_ecg)} windows extracted.")
    return X_ecg, X_eda, X_acc, y, groups

# Allow running this file directly to test it
if __name__ == "__main__":
    print("Running preprocessing pipeline...")
    X_ecg, X_eda, X_acc, y, groups = create_dataset()
    print(f"Final shapes:")
    print(f"  ECG features: {X_ecg.shape}")
    print(f"  EDA features: {X_eda.shape}")
    print(f"  ACC features: {X_acc.shape}")
    print(f"  Labels: {y.shape}")