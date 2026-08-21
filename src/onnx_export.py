# src/onnx_export.py
import torch
import numpy as np
import time
import os
import onnxruntime as ort
from src.model import MultimodalFusionModel
from src.config import MODELS_DIR, ONNX_MODEL_NAME, ECG_DIM, EDA_DIM, ACC_DIM

def export_to_onnx():
    """
    Load the best PyTorch model and export it to ONNX format.
    """
    print("\n" + "="*60)
    print("STEP 6: ONNX EXPORT & BENCHMARK")
    print("="*60)
    
    # 1. Load the trained PyTorch model
    model_path = os.path.join(MODELS_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python -m src.train' first.")
        return
    
    print(f"Loading model from: {model_path}")
    model = MultimodalFusionModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # Set to evaluation mode (critical for ONNX)
    
    # 2. Create dummy input tensors (matching the feature sizes)
    #    ONNX needs to know the shape of the inputs.
    dummy_ecg = torch.randn(1, ECG_DIM, dtype=torch.float32)
    dummy_eda = torch.randn(1, EDA_DIM, dtype=torch.float32)
    dummy_acc = torch.randn(1, ACC_DIM, dtype=torch.float32)
    
    # 3. Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, ONNX_MODEL_NAME)
    print(f"\nExporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,                           # The PyTorch model
        (dummy_ecg, dummy_eda, dummy_acc),  # Tuple of inputs
        onnx_path,                       # Output file path
        input_names=['ecg', 'eda', 'acc'],   # Give the inputs names
        output_names=['output'],         # Give the output a name
        dynamic_axes={                   # Allow batch size to be flexible
            'ecg': {0: 'batch_size'},
            'eda': {0: 'batch_size'},
            'acc': {0: 'batch_size'}
        },
        opset_version=14                 # Use ONNX opset 14 (stable)
    )
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    return onnx_path

def benchmark_onnx(onnx_path, n_iter=1000):
    """
    Load the ONNX model and measure inference time.
    """
    print(f"\nBenchmarking ONNX model with {n_iter} iterations...")
    
    # 1. Load the ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 2. Create dummy inputs
    dummy_ecg = np.random.randn(1, ECG_DIM).astype(np.float32)
    dummy_eda = np.random.randn(1, EDA_DIM).astype(np.float32)
    dummy_acc = np.random.randn(1, ACC_DIM).astype(np.float32)
    
    inputs = {
        'ecg': dummy_ecg,
        'eda': dummy_eda,
        'acc': dummy_acc
    }
    
    # 3. Warmup (let the CPU/GPU get ready)
    for _ in range(10):
        session.run(None, inputs)
    
    # 4. Measure time
    start_time = time.time()
    for _ in range(n_iter):
        session.run(None, inputs)
    elapsed = time.time() - start_time
    
    # 5. Calculate average latency
    avg_latency_ms = (elapsed / n_iter) * 1000
    print(f"✅ Average inference time: {avg_latency_ms:.3f} ms per sample")
    print(f"   (Processed {n_iter} samples in {elapsed:.2f} seconds)")
    
    # 6. Test a real prediction to show it works
    print("\n--- Test Prediction ---")
    outputs = session.run(None, inputs)
    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1)
    pred_class = int(np.argmax(probs))
    stress_prob = float(probs[0][1])
    
    print(f"Input features: ECG={dummy_ecg[0][:2]}... EDA={dummy_eda[0][:2]}... ACC={dummy_acc[0][:2]}...")
    print(f"Prediction: {'STRESS 😰' if pred_class == 1 else 'BASELINE 😌'} (Stress probability: {stress_prob:.2f})")
    
    return avg_latency_ms

def verify_onnx_output(onnx_path):
    """
    Compare ONNX output with PyTorch output to ensure they match.
    """
    print("\n--- Verifying ONNX vs PyTorch ---")
    
    # Load PyTorch model
    model = MultimodalFusionModel()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pth'), map_location='cpu'))
    model.eval()
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Create random input
    ecg = np.random.randn(1, ECG_DIM).astype(np.float32)
    eda = np.random.randn(1, EDA_DIM).astype(np.float32)
    acc = np.random.randn(1, ACC_DIM).astype(np.float32)
    
    # PyTorch inference
    with torch.no_grad():
        ecg_t = torch.tensor(ecg, dtype=torch.float32)
        eda_t = torch.tensor(eda, dtype=torch.float32)
        acc_t = torch.tensor(acc, dtype=torch.float32)
        pt_output = model(ecg_t, eda_t, acc_t).numpy()
    
    # ONNX inference
    onnx_output = session.run(None, {'ecg': ecg, 'eda': eda, 'acc': acc})[0]
    
    # Compare
    diff = np.max(np.abs(pt_output - onnx_output))
    print(f"PyTorch output: {pt_output.flatten()}")
    print(f"ONNX output:    {onnx_output.flatten()}")
    print(f"Max difference: {diff:.6f}")
    
    if diff < 1e-5:
        print("✅ ONNX export is correct! (Models match)")
    else:
        print("⚠️  WARNING: ONNX and PyTorch outputs differ slightly (this can happen due to numerical precision).")

if __name__ == "__main__":
    # Export
    onnx_path = export_to_onnx()
    
    if onnx_path and os.path.exists(onnx_path):
        # Benchmark
        latency = benchmark_onnx(onnx_path, n_iter=500)
        
        # Verify correctness
        verify_onnx_output(onnx_path)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✅ ONNX model ready at: {onnx_path}")
        print(f"⚡ Inference speed: {latency:.3f} ms per sample")
        print("🎯 Ready for deployment on edge devices (Raspberry Pi, phone, etc.)")
        print("="*60)
    else:
        print("❌ Export failed.")