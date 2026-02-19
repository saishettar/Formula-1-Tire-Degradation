"""
src/features.py
---------------
Feature engineering utilities.

Mirrors the notebook baseline:
- Numeric: Tire_wear, Humidity, Ambient_Temperature
- Categorical: Event -> factorized integer code (Event_enc)

If you want a stronger encoding later (OneHotEncoder), swap the encoder here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_FEATURES = ["Tire_wear", "Humidity", "Ambient_Temperature", "Event"]
DEFAULT_NUMERIC_FEATURES = ["Tire_wear", "Humidity", "Ambient_Temperature"]
DEFAULT_CAT_FEATURE = "Event"


@dataclass
class FeatureArtifacts:
    event_categories: List[str]

    def to_dict(self) -> Dict:
        return {"event_categories": self.event_categories}

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureArtifacts":
        return cls(event_categories=list(d["event_categories"]))


def fit_event_factorizer(series: pd.Series) -> Tuple[pd.Series, List[str]]:
    s = series.astype(str).fillna("Unknown")
    enc, categories = pd.factorize(s, sort=True)
    return pd.Series(enc, index=series.index, name="Event_enc"), list(categories)


def transform_event_factorizer(series: pd.Series, categories: List[str]) -> pd.Series:
    s = series.astype(str).fillna("Unknown")
    cat_to_id = {c: i for i, c in enumerate(categories)}
    # unseen events map to -1
    enc = s.map(cat_to_id).fillna(-1).astype(int)
    return pd.Series(enc, index=series.index, name="Event_enc")


def build_features(
    df: pd.DataFrame,
    *,
    target: str = "Tire_Degradation",
    fit: bool = True,
    artifacts: FeatureArtifacts | None = None,
) -> Tuple[pd.DataFrame, pd.Series, FeatureArtifacts]:
    """
    Returns X, y, and feature artifacts (event categories mapping).
    """
    missing = [c for c in DEFAULT_FEATURES + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if fit:
        event_enc, categories = fit_event_factorizer(df[DEFAULT_CAT_FEATURE])
        artifacts = FeatureArtifacts(event_categories=categories)
    else:
        if artifacts is None:
            raise ValueError("artifacts must be provided when fit=False")
        event_enc = transform_event_factorizer(df[DEFAULT_CAT_FEATURE], artifacts.event_categories)

    X = df[DEFAULT_NUMERIC_FEATURES].copy()
    X["Event_enc"] = event_enc
    y = df[target].copy()
    return X, y, artifacts
