import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

st.set_page_config(page_title="EV Range Predictor", layout="centered")

st.title("⚡ EV Electric Range Predictor (Random Forest)")
st.write("Enter key values below. The app will predict **electric_range** using your saved Random Forest model.")

# ----------------------------
# Helpers: find latest files
# ----------------------------
def find_latest(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

MODEL_PATH = find_latest("best_random_forest_model_*.pkl") or find_latest("best_random_forest_model*.pkl")
META_PATH  = find_latest("model_metadata_*.pkl") or find_latest("model_metadata*.pkl")

st.sidebar.header("📦 Loaded Files")
st.sidebar.write(f"Model: `{MODEL_PATH}`" if MODEL_PATH else "Model: ❌ Not found")
st.sidebar.write(f"Metadata: `{META_PATH}`" if META_PATH else "Metadata: ❌ Not found")
st.sidebar.caption(f"scikit-learn: {sklearn.__version__}")

if (MODEL_PATH is None) or (META_PATH is None):
    st.error("I couldn't find your model or metadata file in this folder. Make sure both .pkl files are inside EV_Streamlit_Demo.")
    st.stop()

# ----------------------------
# Load model + metadata
# ----------------------------
try:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)
except Exception as e:
    st.error("Failed to load model/metadata. See error below:")
    st.code(str(e))
    st.stop()

# We expect metadata to contain feature columns
if isinstance(metadata, dict) and "feature_names" in metadata:
    feature_cols = metadata["feature_names"]
else:
    st.error("Could not find 'feature_names' inside metadata.")
    st.stop()

st.sidebar.caption(f"Total expected features: {len(feature_cols):,}")

# ----------------------------
# Helpers: one-hot selection
# ----------------------------
def onehot_options(prefix: str, cols):
    return sorted([c for c in cols if c.startswith(prefix)])

def set_onehot(X, chosen_col, options):
    for c in options:
        X.loc[0, c] = 0
    if chosen_col in options:
        X.loc[0, chosen_col] = 1

# ----------------------------
# Inputs
# ----------------------------
st.subheader("🧾 Inputs")

vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5)
model_year  = st.number_input("Model Year", min_value=1990, max_value=2035, value=2020)

st.info("Note: Base MSRP was removed because the trained model does not include it as a feature.")

# Detect common one-hot groups (adjust prefixes if your feature names differ)
county_opts   = onehot_options("county_", feature_cols)
veh_type_opts = onehot_options("vehicle_type_", feature_cols)
cafv_opts     = onehot_options("cafv_eligibility_", feature_cols)

st.caption("To make predictions realistic, I select key categorical fields so the correct one-hot feature is activated.")

selected_county = st.selectbox("County", county_opts) if county_opts else None
selected_vehicle_type = st.selectbox("Vehicle Type", veh_type_opts) if veh_type_opts else None
selected_cafv = st.selectbox("CAFV Eligibility", cafv_opts) if cafv_opts else None

# ----------------------------
# Build prediction row
# ----------------------------
X_input = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)

# Set numeric columns (only if they exist)
if "vehicle_age" in X_input.columns:
    X_input.loc[0, "vehicle_age"] = vehicle_age
if "model_year" in X_input.columns:
    X_input.loc[0, "model_year"] = model_year

# Set one-hot categoricals (only if detected)
if county_opts and selected_county:
    set_onehot(X_input, selected_county, county_opts)
if veh_type_opts and selected_vehicle_type:
    set_onehot(X_input, selected_vehicle_type, veh_type_opts)
if cafv_opts and selected_cafv:
    set_onehot(X_input, selected_cafv, cafv_opts)

# ----------------------------
# Predict
# ----------------------------
if st.button("🔮 Predict Electric Range"):
    try:
        pred = model.predict(X_input)[0]
        st.success(f"✅ Predicted Electric Range: **{pred:.2f} miles**")
    except Exception as e:
        st.error("Prediction failed. Error:")
        st.code(str(e))

with st.expander("🔍 Show input vector (first 30 columns)"):
    st.dataframe(X_input.iloc[:, :30])