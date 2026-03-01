import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EV Range Predictor", layout="centered")

st.title("⚡ EV Electric Range Predictor (Random Forest)")
st.write(
    "Enter values below. The app will predict **electric_range** using your saved Random Forest model."
)

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

if (MODEL_PATH is None) or (META_PATH is None):
    st.error(
        "I couldn't find your model or metadata file in this folder. "
        "Make sure both .pkl files are inside EV_Streamlit_Demo."
    )
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

if feature_cols is None:
    st.error("Metadata loaded, but I couldn't find the feature column list inside it.")
    st.write("Metadata keys found:", list(metadata.keys()) if isinstance(metadata, dict) else type(metadata))
    st.stop()

# ----------------------------
# Detect one-hot groups by prefix
# ----------------------------
def get_onehot_options(prefix: str):
    return [c for c in feature_cols if c.startswith(prefix)]

county_options = get_onehot_options("county_")
make_options   = get_onehot_options("make_")
model_options  = get_onehot_options("model_")
type_options   = get_onehot_options("vehicle_type_")   # change if your prefix differs
cafv_options   = get_onehot_options("cafv_")           # change if your prefix differs

# ----------------------------
# Inputs
# ----------------------------
st.subheader("🧾 Numeric Inputs")

vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5)
model_year  = st.number_input("Model Year", min_value=1990, max_value=2035, value=2020)

# MSRP handling: show only if model actually expects it
if "base_msrp" in feature_cols:
    base_msrp = st.number_input("Base MSRP", min_value=0, max_value=500000, value=40000, step=500)
else:
    base_msrp = None
    st.info("Note: Base MSRP was removed because the trained model does not include it as a feature.")

st.caption("To make predictions realistic, I select key categorical fields so the correct one-hot feature is activated.")

st.subheader("📌 Categorical Selections (One-Hot)")

selected_county = st.selectbox("County", county_options) if county_options else None
selected_make   = st.selectbox("Make", make_options) if make_options else None
selected_model  = st.selectbox("Model", model_options) if model_options else None
selected_type   = st.selectbox("Vehicle Type", type_options) if type_options else None
selected_cafv   = st.selectbox("CAFV Eligibility", cafv_options) if cafv_options else None

# ----------------------------
# Build prediction row
# ----------------------------
X_input = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)

# Set numeric fields only if they exist
if "vehicle_age" in X_input.columns:
    X_input.loc[0, "vehicle_age"] = vehicle_age
if "model_year" in X_input.columns:
    X_input.loc[0, "model_year"] = model_year
if base_msrp is not None and "base_msrp" in X_input.columns:
    X_input.loc[0, "base_msrp"] = base_msrp

# Activate selected one-hot columns (set them to 1)
for selected in [selected_county, selected_make, selected_model, selected_type, selected_cafv]:
    if selected is not None and selected in X_input.columns:
        X_input.loc[0, selected] = 1

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

with st.expander("🧩 Debug: Show detected one-hot groups"):
    st.write("county_ options:", len(county_options))
    st.write("make_ options:", len(make_options))
    st.write("model_ options:", len(model_options))
    st.write("vehicle_type_ options:", len(type_options))
    st.write("cafv_ options:", len(cafv_options))