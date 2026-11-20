# Credit Card Fraud Detection

Comparing 5 machine learning models on an extremely imbalanced dataset (0.2% fraud rate).

## Problem

Standard accuracy is misleading on imbalanced data.
A model predicting everything as legit gets 99.8% accuracy but catches zero fraud.
Primary metric used: **PR-AUC** (Precision-Recall AUC), since ROC-AUC is overly optimistic when the positive class is this rare.

## What I Did

- 80/10/10 train/val/test split
- StandardScaler fit on train only, applied to Time and Amount
- SMOTE applied only on training data to handle class imbalance — val/test are left untouched to stay realistic
- Compared 5 models: Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression
- Model selection done on the validation set only; test set evaluated once, at the end, for an unbiased final read

## Results (test set)

| Model          | Precision | Recall | F1   | PR-AUC |
|----------------|-----------|--------|------|--------|
| Random Forest  | 0.85      | 0.81   | 0.83 | 0.87   |
| XGBoost        | 0.77      | 0.84   | 0.80 | 0.87   |
| CatBoost       | 0.55      | 0.86   | 0.67 | 0.84   |
| LightGBM       | 0.77      | 0.84   | 0.80 | 0.79   |
| Logistic Reg   | 0.05      | 0.91   | 0.10 | 0.77   |

**Winner: Random Forest**, selected on the validation set (PR-AUC 0.903). On the held-out test set it holds up well (PR-AUC 0.867), with XGBoost essentially tied at 0.866 — the two models swap first place by margins of 0.001–0.002 depending on the split. Both are reported above for transparency.

## Key Findings

- Tree-based and boosted models significantly outperform Logistic Regression — confirms fraud patterns are non-linear.
- **Random Forest and XGBoost are essentially tied.** Across multiple validation/test splits, the two models swap first place by margins of 0.002–0.006 PR-AUC. Neither is reliably "better" — pick either with confidence.
- **CatBoost has a real, reproducible precision problem.** Recall stays strong (~0.86), but precision repeatedly lands around 0.52–0.56 across every split tested. Inspecting the probability distribution for true fraud cases showed scores clustering near 0 or 1, not in the adjustable middle — so the issue isn't a misplaced default threshold, it's the model's confidence calibration itself.
- **Threshold tuning was tested and didn't help.** Given the above, lowering or raising the 0.5 cutoff had little effect, since there were few borderline-probability cases to move across the boundary. This is a case where the standard fix didn't apply — worth knowing rather than assuming threshold tuning is always the answer.
- Logistic Regression's precision collapses to ~0.05–0.07 despite high recall, confirming a linear decision boundary is a poor fit for this problem.

## Stack

Python, Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn, FastAPI, Uvicorn, Pydantic

## Dataset

This repo does not include the dataset due to file size.

**Download:** [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Setup:**
1. Download `creditcard.csv` from the Kaggle link above (requires a free Kaggle account).
2. Place `creditcard.csv` in the same folder as the project files.
3. Install dependencies:
   ```
   pip install pandas numpy scikit-learn imbalanced-learn catboost xgboost lightgbm joblib fastapi uvicorn pydantic
   ```
4. Train the model — this saves `model.pkl` and `scaler.pkl`:
   ```
   python fraud_detection.py
   ```
   Validation results, the selected winner, and final test results are printed to the console.

5. Start the API:
   ```
   uvicorn main:app --reload
   ```

## API

The API is served by FastAPI. Once running, interactive docs are available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| POST | `/predict` | Run fraud prediction on a transaction |
| POST | `/report/{record_id}` | Report whether fraud actually occurred |
| GET | `/accuracy` | Brier score across all resolved predictions |
| GET | `/drift` | Compare recent vs previous Brier score to detect model degradation |

Predictions are logged to `predictions.json`. Once outcomes are reported via `/report`, accuracy and drift endpoints become meaningful.

To populate enough data for drift detection (requires 400 resolved predictions), run the smoke test:
```
python smoke_test.py
```
