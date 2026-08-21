# src/dashboard.py
import sys
import os
# Add project root to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from src.config import ECG_DIM, EDA_DIM, ACC_DIM

# ---------------------------
# 1. Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Stress Inference Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# 2. Title and Description
# ---------------------------
st.title("🧠 Multimodal Stress Inference")
st.markdown("""
This app uses a **multimodal neural network** to predict stress levels from:
- **ECG** (heart rate variability features)
- **EDA** (skin conductance features)
- **ACC** (movement features)

Adjust the sliders below and click **Predict** to see the result.
""")

# ---------------------------
# 3. Sidebar: API Configuration
# ---------------------------
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. The dashboard sends features to the API.
2. The API runs the ONNX model.
3. The prediction is displayed here.

Make sure the API is running (`python -m src.api`).
""")

# ---------------------------
# 4. Main Area: Input Sliders
# ---------------------------
st.subheader("📊 Input Features")

# Create three columns for ECG, EDA, ACC
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ❤️ ECG Features (HRV)")
    ecg_values = []
    ecg_names = ["MeanNN", "SDNN", "RMSSD", "pNN50", "MeanHR"]
    for i, name in enumerate(ecg_names):
        val = st.slider(
            f"{name}",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            key=f"ecg_{i}"
        )
        ecg_values.append(val)

with col2:
    st.markdown("### 💧 EDA Features")
    eda_values = []
    eda_names = ["Mean", "Std", "Range", "Tonic", "SCR Count"]
    for i, name in enumerate(eda_names):
        val = st.slider(
            f"{name}",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            key=f"eda_{i}"
        )
        eda_values.append(val)

with col3:
    st.markdown("### 📳 ACC Features")
    acc_values = []
    acc_names = ["Mean X", "Mean Y", "Mean Z", "Std X", "Std Y", "Std Z", "Mean Mag", "Std Mag"]
    for i, name in enumerate(acc_names):
        val = st.slider(
            f"{name}",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            key=f"acc_{i}"
        )
        acc_values.append(val)

# ---------------------------
# 5. Predict Button
# ---------------------------
st.markdown("---")
col_btn, col_spacer = st.columns([1, 3])

with col_btn:
    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# ---------------------------
# 6. Make Prediction
# ---------------------------
if predict_clicked:
    # Build the request payload
    payload = {
        "ecg": ecg_values,
        "eda": eda_values,
        "acc": acc_values
    }
    
    # Show a spinner while waiting
    with st.spinner("Calling API..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json=payload,
                timeout=10
            )
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Make sure the server is running.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("❌ API request timed out.")
            st.stop()
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        
        # Display results
        st.success("✅ Prediction received!")
        
        # Create two columns for results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("📋 Prediction")
            pred_label = result["label"]
            confidence = result["confidence"]
            
            # Color the prediction based on result
            if pred_label.startswith("STRESS"):
                st.markdown(f"## 😰 **{pred_label}**")
            else:
                st.markdown(f"## 😌 **{pred_label}**")
            
            st.metric("Confidence", f"{confidence:.2%}")
        
        with res_col2:
            st.subheader("📊 Probabilities")
            
            # Create a bar chart
            probs = result["probabilities"]
            fig, ax = plt.subplots(figsize=(6, 3))
            categories = ["Baseline", "Stress"]
            values = [probs["baseline"], probs["stress"]]
            colors = ["#2E86AB", "#F18F01"]
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities")
            
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{val:.2%}", ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        # Show input summary
        with st.expander("📥 Input Summary"):
            st.json({
                "ecg": ecg_values,
                "eda": eda_values,
                "acc": acc_values
            })
    
    else:
        st.error(f"❌ API error: {response.status_code}")
        st.text(response.text)

# ---------------------------
# 7. Footer
# ---------------------------
st.markdown("---")
st.markdown("""
**Built with:** Streamlit · FastAPI · PyTorch · ONNX
""")