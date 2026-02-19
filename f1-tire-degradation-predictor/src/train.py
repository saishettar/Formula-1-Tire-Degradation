"""
src/train.py
------------
Train and evaluate multiple regressors, then save the best model.

Usage:
    python -m src.train --csv_path data/simulated_dataset.csv
    python -m src.train --use_kagglehub
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from .preprocess import DEFAULT_TARGET, load_from_csv, load_from_kagglehub
from .features import build_features

RANDOM_STATE = 42


def evaluate(name: str, model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    return {
        "model": name,
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }


def train_all(X_train, y_train) -> Dict[str, object]:
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            random_state=RANDOM_STATE,
            max_iter=800,
            early_stopping=True,
        ),
    }
    fitted: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def select_best(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    items = list(metrics.items())
    items.sort(key=lambda kv: (-kv[1]["r2"], kv[1]["rmse"]))
    return items[0][0], items[0][1]


def main() -> None:
    p = argparse.ArgumentParser(description="Train tire degradation models and save the best.")
    p.add_argument("--csv_path", type=str, default=None, help="Path to local CSV dataset.")
    p.add_argument("--use_kagglehub", action="store_true", help="Load dataset via kagglehub.")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--out_dir", type=str, default="outputs")
    args = p.parse_args()

    if args.use_kagglehub:
        df = load_from_kagglehub(target=args.target)
    else:
        if not args.csv_path:
            raise SystemExit("Provide --csv_path or pass --use_kagglehub")
        df = load_from_csv(args.csv_path, target=args.target)

    X, y, artifacts = build_features(df, target=args.target, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    fitted = train_all(X_train, y_train)

    metrics: Dict[str, Dict[str, float]] = {}
    for name, model in fitted.items():
        metrics[name] = evaluate(name, model, X_test, y_test)

    best_name, best_metrics = select_best(metrics)
    best_model = fitted[best_name]

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.joblib")
    dump(best_model, model_path)

    artifacts_path = os.path.join(args.out_dir, "feature_artifacts.json")
    with open(artifacts_path, "w") as f:
        json.dump(artifacts.to_dict(), f, indent=2)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"best_model": best_name, "best_metrics": best_metrics, "all_metrics": metrics},
            f,
            indent=2,
        )

    print("Training complete.")
    print(f"Best model: {best_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved artifacts to: {artifacts_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
