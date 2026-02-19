"""
src/predict.py
--------------
Load a saved model and run predictions.

Usage:
    python -m src.predict --input_csv data/sample_input.csv --out_csv outputs/preds.csv
    python -m src.predict --tire_wear 0.42 --humidity 55 --ambient_temperature 22 --event "Monaco GP"
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from joblib import load

from .features import build_features, FeatureArtifacts


def load_artifacts(out_dir: str) -> FeatureArtifacts:
    path = os.path.join(out_dir, "feature_artifacts.json")
    with open(path, "r") as f:
        d = json.load(f)
    return FeatureArtifacts.from_dict(d)


def main() -> None:
    p = argparse.ArgumentParser(description="Predict tire degradation using a saved model.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Directory with model.joblib and artifacts.")
    p.add_argument("--target", type=str, default="Tire_Degradation")

    p.add_argument("--input_csv", type=str, default=None, help="CSV with Tire_wear, Humidity, Ambient_Temperature, Event")
    p.add_argument("--out_csv", type=str, default=None)

    p.add_argument("--tire_wear", type=float, default=None)
    p.add_argument("--humidity", type=float, default=None)
    p.add_argument("--ambient_temperature", type=float, default=None)
    p.add_argument("--event", type=str, default=None)
    args = p.parse_args()

    model_path = os.path.join(args.out_dir, "model.joblib")
    model = load(model_path)
    artifacts = load_artifacts(args.out_dir)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        df2 = df.copy()
        df2[args.target] = 0.0
        X, _, _ = build_features(df2, target=args.target, fit=False, artifacts=artifacts)
        preds = model.predict(X)

        out = df.copy()
        out["predicted_tire_degradation"] = preds
        print(out.head(20).to_string(index=False))

        if args.out_csv:
            out.to_csv(args.out_csv, index=False)
            print(f"\nWrote predictions to: {args.out_csv}")
        return

    required = [args.tire_wear, args.humidity, args.ambient_temperature, args.event]
    if any(v is None for v in required):
        raise SystemExit("Provide --input_csv OR all of: --tire_wear --humidity --ambient_temperature --event")

    df = pd.DataFrame([{
        "Tire_wear": args.tire_wear,
        "Humidity": args.humidity,
        "Ambient_Temperature": args.ambient_temperature,
        "Event": args.event,
        args.target: 0.0,
    }])

    X, _, _ = build_features(df, target=args.target, fit=False, artifacts=artifacts)
    pred = float(model.predict(X)[0])
    print(f"Predicted Tire Degradation: {pred:.6f}")


if __name__ == "__main__":
    main()
