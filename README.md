# Kaggle Quantitative Forecasting (iRage Challenge) — v6.1 Pipeline

This repository contains my solution pipeline for the **Short-Horizon Return Prediction Challenge by iRage** (Kaggle/hackathon setting).

The training code implements a feature-engineering + clustering + regime-aware ensembling approach, with GPU acceleration via XGBoost Random Forest.

---

## 🧠 Approach Summary

### Core ideas (v6.0 → v6.1)

1. **Orthogonal momentum windows** from lag features:
   - `W1 = LagT1`
   - `W2 = LagT2 - LagT1`
   - `W3 = LagT3 - LagT2`

2. **Drop all Lag0/base features** to reduce non-stationarity risk.

3. **Cluster transformed W-features** using hierarchical clustering on Spearman correlation distance.

4. **Select one representative per cluster by variance**  
   (target-independent selection to avoid look-ahead leakage in feature selection).

5. **Regime encoding with `SO3_T`**:
   - Auto-band `SO3_T` into quantile bins
   - One-hot encode bands for regime isolation

6. **Volatility-aware target scaling**:
   - Scale target by per-regime std during training
   - Re-scale predictions back during inference

7. **Chronological validation** using `GroupKFold` over `CV_GROUP`.

8. **3-model weighted ensemble**:
   - 40% Ridge
   - 40% XGBRFRegressor
   - 20% Huber LightGBM

### v6.1 specific upgrade

- Replaced `sklearn.RandomForestRegressor` with **`xgboost.XGBRFRegressor`**
- Enabled GPU acceleration (`tree_method='hist'`, `device='cuda'`)
- Added stronger RF-style randomness (`colsample_bynode=0.15`)
- Reduced training runtime significantly vs prior CPU version

---

## ⚙️ Pipeline Steps

1. Load train/test parquet
2. Identify features having complete lag set (`LagT1`, `LagT2`, `LagT3`)
3. Build orthogonal windows (`W1`, `W2`, `W3`)
4. Cluster W-features by absolute Spearman correlation distance
5. Select high-variance representative from each cluster
6. Detect `SO3_T` bands and one-hot encode
7. Compute regime statistics on target
8. Scale target by regime volatility
9. Train ensemble with `GroupKFold`
10. Predict test set and inverse-scale by regime std
11. Export submission CSV

---

## 🗂️ Expected Data Layout

When running on Kaggle (`IS_KAGGLE = True`):

- `/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/train.parquet`
- `/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/test.parquet`

When running locally (`IS_KAGGLE = False`), update:

- `DATASET/short-horizon-return-prediction-challenge-by-i-rage/train-001.parquet`
- `DATASET/short-horizon-return-prediction-challenge-by-i-rage/test.parquet`

Required columns include:
- `ID`
- `TARGET` (train only)
- `CV_GROUP` (train only)
- `SO3_T`
- lagged feature columns (`*_LagT1`, `*_LagT2`, `*_LagT3`)

---

## 🔧 Dependencies

- Python 3.9+
- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- lightgbm
- pyarrow (for parquet IO)

Install:

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm pyarrow
```

---

## ▶️ How to Run

1. Place data files in the configured path.
2. Set `IS_KAGGLE` appropriately in the script.
3. Run:

```bash
python train_v6_1.py
```

Output:
- `submission_v6.1.csv`

---

## 📊 Validation Strategy

- `GroupKFold(n_splits=5)` with `CV_GROUP` to preserve temporal grouping structure and reduce leakage risk from random splits.

- Reported during training:
  - Fold-wise R² for each base model
  - Overall OOF R² for Ridge, XGB-RF, LightGBM, and ensemble

---

## 🧪 Notes / Design Choices

- Feature clustering uses **absolute** Spearman correlation (`1 - |corr|`) so both positive and negative co-movements are grouped.
- Cluster representative is selected by **variance**, not target correlation, to keep feature selection target-agnostic.
- Regime scaling normalizes learning across volatility states and is reversed at inference time.

---

## ⚠️ Reproducibility Notes

- Seed is set where supported (`random_state=42`).
- Exact scores may vary slightly by:
  - GPU/driver versions
  - XGBoost/LightGBM versions
  - Numeric precision/hardware behavior

---

## 📌 Repository Context

This repository is a past hackathon/Kaggle competition solution and is shared for learning, reproducibility, and experimentation purposes.

If you reuse the idea, adapt CV and leakage controls carefully for your own competition/problem setup.
