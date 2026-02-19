import streamlit as st
import pandas as pd
import os
from joblib import load
import json

from src.features import build_features, FeatureArtifacts

st.set_page_config(page_title="F1 Tire Degradation Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ F1 Tire Degradation Predictor")
st.write("Enter conditions and get a predicted tire degradation value.")

out_dir = st.sidebar.text_input("Model folder", value="outputs")
model_path = os.path.join(out_dir, "model.joblib")
artifacts_path = os.path.join(out_dir, "feature_artifacts.json")

if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
    st.warning("Model not found. Train first: `python -m src.train --csv_path <path>`")
    st.stop()

model = load(model_path)
with open(artifacts_path, "r") as f:
    artifacts = FeatureArtifacts.from_dict(json.load(f))

tire_wear = st.slider("Tire wear", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
ambient_temp = st.number_input("Ambient temperature (°C)", min_value=-10.0, max_value=60.0, value=22.0, step=0.5)
event_default = artifacts.event_categories[0] if artifacts.event_categories else "Unknown"
event = st.text_input("Event", value=event_default)

df = pd.DataFrame([{
    "Tire_wear": tire_wear,
    "Humidity": humidity,
    "Ambient_Temperature": ambient_temp,
    "Event": event,
    "Tire_Degradation": 0.0,
}])

X, _, _ = build_features(df, target="Tire_Degradation", fit=False, artifacts=artifacts)
pred = float(model.predict(X)[0])

st.metric("Predicted Tire Degradation", f"{pred:.4f}")
