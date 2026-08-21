# src/config.py
import os

# ---------------------------
# 1. Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------
# 2. WESAD subjects
# ---------------------------
SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]

# ---------------------------
# 3. Signal parameters
# ---------------------------
FS = 700
WINDOW_SEC = 60
STEP_SEC = 30

# ---------------------------
# 4. Feature dimensions
# ---------------------------
ECG_DIM = 5
EDA_DIM = 5
ACC_DIM = 8

# ---------------------------
# 5. Training hyperparameters
# ---------------------------
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_FUSION = 64

# ---------------------------
# 6. ONNX model name
# ---------------------------
ONNX_MODEL_NAME = "multimodal_model.onnx"