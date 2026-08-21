# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ECG_DIM, EDA_DIM, ACC_DIM, HIDDEN_FUSION

class ModalityEncoder(nn.Module):
    """
    A small neural network that encodes ONE modality (ECG, EDA, or ACC)
    into a 32‑dimension vector.
    
    Input:  (batch_size, feature_dim)  – e.g., (64, 5) for ECG
    Output: (batch_size, 32)
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)  # no activation here – we'll add it after fusion
        return x

class MultimodalFusionModel(nn.Module):
    """
    Takes ECG, EDA, and ACC features, encodes each separately,
    then fuses them and outputs class probabilities.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Three separate encoders – one for each modality
        self.ecg_enc = ModalityEncoder(ECG_DIM)
        self.eda_enc = ModalityEncoder(EDA_DIM)
        self.acc_enc = ModalityEncoder(ACC_DIM)
        
        # Fusion layer: takes 3 x 32 = 96 concatenated features
        fusion_dim = 32 * 3
        self.fc_fusion = nn.Linear(fusion_dim, HIDDEN_FUSION)
        self.bn_fusion = nn.BatchNorm1d(HIDDEN_FUSION)
        
        # Final output layer
        self.fc_out = nn.Linear(HIDDEN_FUSION, num_classes)
    
    def forward(self, x_ecg, x_eda, x_acc):
        # 1. Encode each modality
        z_ecg = self.ecg_enc(x_ecg)   # (batch, 32)
        z_eda = self.eda_enc(x_eda)   # (batch, 32)
        z_acc = self.acc_enc(x_acc)   # (batch, 32)
        
        # 2. Concatenate (fuse) them
        z = torch.cat([z_ecg, z_eda, z_acc], dim=1)  # (batch, 96)
        
        # 3. Pass through fusion layer
        z = F.relu(self.bn_fusion(self.fc_fusion(z)))  # (batch, HIDDEN_FUSION)
        
        # 4. Output logits (raw scores before softmax)
        logits = self.fc_out(z)  # (batch, 2)
        return logits