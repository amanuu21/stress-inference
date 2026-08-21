# src/api.py
import os
import numpy as np
import onnxruntime as ort
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import MODELS_DIR, ONNX_MODEL_NAME, ECG_DIM, EDA_DIM, ACC_DIM

# ---------------------------
# 1. Define the input schema
# ---------------------------
class Features(BaseModel):
    """
    This is the shape of data the API expects.
    FastAPI automatically validates incoming JSON against this model.
    """
    ecg: list  # List of 5 numbers (HRV features)
    eda: list  # List of 5 numbers (EDA features)
    acc: list  # List of 8 numbers (ACC features)

# ---------------------------
# 2. Load the ONNX model and scalers
# ---------------------------
print("Loading ONNX model...")
onnx_path = os.path.join(MODELS_DIR, ONNX_MODEL_NAME)
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Run 'python -m src.onnx_export' first.")

# ONNX Runtime session
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# Load scalers (used to normalize input features the same way as training)
print("Loading scalers...")
scaler_ecg = joblib.load(os.path.join(MODELS_DIR, 'scaler_ecg.pkl'))
scaler_eda = joblib.load(os.path.join(MODELS_DIR, 'scaler_eda.pkl'))
scaler_acc = joblib.load(os.path.join(MODELS_DIR, 'scaler_acc.pkl'))

print("✅ API ready!")

# ---------------------------
# 3. Create the FastAPI app
# ---------------------------
app = FastAPI(
    title="Stress Inference API",
    description="Predict stress level from ECG, EDA, and ACC features.",
    version="1.0.0"
)

@app.get("/")
def root():
    """Root endpoint – just a welcome message."""
    return {
        "message": "Stress Inference API is running!",
        "endpoints": {
            "/predict": "POST – send features for prediction",
            "/health": "GET – check if the server is alive"
        },
        "docs": "/docs"  # FastAPI auto-generates interactive docs
    }

@app.get("/health")
def health():
    """Health check endpoint – confirms the server is running."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(features: Features):
    """
    Predict stress level from ECG, EDA, and ACC features.
    
    Input: JSON with ecg (list of 5), eda (list of 5), acc (list of 8)
    Output: Prediction (0=baseline, 1=stress) with probability.
    """
    # 1. Validate input lengths
    if len(features.ecg) != ECG_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"ECG must have {ECG_DIM} features, got {len(features.ecg)}"
        )
    if len(features.eda) != EDA_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"EDA must have {EDA_DIM} features, got {len(features.eda)}"
        )
    if len(features.acc) != ACC_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"ACC must have {ACC_DIM} features, got {len(features.acc)}"
        )
    
    # 2. Convert to numpy and reshape (1 sample, N features)
    ecg = np.array(features.ecg, dtype=np.float32).reshape(1, -1)
    eda = np.array(features.eda, dtype=np.float32).reshape(1, -1)
    acc = np.array(features.acc, dtype=np.float32).reshape(1, -1)
    
    # 3. Scale the features (using the same scalers from training)
    ecg = scaler_ecg.transform(ecg)
    eda = scaler_eda.transform(eda)
    acc = scaler_acc.transform(acc)
    
    # 4. Run ONNX inference
    outputs = session.run(
        None,  # None = return all outputs
        {
            'ecg': ecg,
            'eda': eda,
            'acc': acc
        }
    )
    
    # 5. Convert logits to probabilities
    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1)
    
    # 6. Extract prediction
    pred_class = int(np.argmax(probs))
    stress_prob = float(probs[0][1])
    baseline_prob = float(probs[0][0])
    
    # 7. Return results
    return {
        "prediction": pred_class,
        "label": "STRESS 😰" if pred_class == 1 else "BASELINE 😌",
        "confidence": float(probs[0][pred_class]),
        "probabilities": {
            "baseline": baseline_prob,
            "stress": stress_prob
        },
        "input_summary": {
            "ecg_features": features.ecg,
            "eda_features": features.eda,
            "acc_features": features.acc
        }
    }

# ---------------------------
# 4. Run the server (if this file is executed directly)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("STARTING STRESS INFERENCE API")
    print("="*60)
    print("Server will run at: http://localhost:8000")
    print("Interactive docs:   http://localhost:8000/docs")
    print("Health check:       http://localhost:8000/health")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)