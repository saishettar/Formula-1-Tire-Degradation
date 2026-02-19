# F1 Tire Degradation Predictor (Machine Learning)

This repo packages your original notebook into a clean, reproducible ML pipeline:

- **Notebook / EDA:** `notebooks/modeling.ipynb`
- **Training:** `src/train.py`
- **Prediction:** `src/predict.py`
- **(Optional) Demo app:** `app/app.py` (Streamlit)

## Structure

```
data/               # put your dataset here (keep full dataset out of git)
notebooks/          # EDA + experimentation
src/                # reproducible pipeline (train/predict)
outputs/            # saved model + artifacts (generated)
app/                # optional Streamlit demo
```

## Dataset

The notebook uses the Kaggle dataset:
- `samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset`
- file: `simulated_dataset.csv`

You can either:
1) Download the CSV and place it in `data/`, then train from local CSV (recommended), or  
2) Load via `kagglehub` (requires Kaggle credentials + internet).

## Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train

**From local CSV**
```bash
python -m src.train --csv_path data/simulated_dataset.csv
```

**Or from kagglehub**
```bash
python -m src.train --use_kagglehub
```

Artifacts written to `outputs/`:
- `model.joblib`
- `feature_artifacts.json`
- `metrics.json`

### Predict

**Single prediction**
```bash
python -m src.predict --tire_wear 0.42 --humidity 55 --ambient_temperature 22 --event "Monaco GP"
```

**Batch prediction**
```bash
python -m src.predict --input_csv data/sample_input.csv --out_csv outputs/preds.csv
```

## Demo app (optional)

```bash
streamlit run app/app.py
```
