import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegressionCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

def print_metrics(row):
    print(f"  Precision: {row['Precision']:.3f}")
    print(f"  Recall:    {row['Recall']:.3f}")
    print(f"  F1:        {row['F1']:.3f}")
    print(f"  PR-AUC:    {row['PR-AUC']:.3f}")

rf_params = {
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs': -1
}

xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42,
    'n_jobs': -1
}

lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'objective': 'binary',
}

cat_params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'random_state': 42,
    'verbose': 0
}

lr_params = {
    'max_iter': 1000,
    'cv': 5,
    'random_state': 42,
    'n_jobs': -1,
    'l1_ratios': (0.0,),              
    'use_legacy_attributes': False    
}


df = pd.read_csv("creditcard.csv")


# reduce memory usage
v_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
df[v_cols] = df[v_cols].astype('float32') 
df["Class"] = df["Class"].astype("int32")


X = df.drop('Class', axis=1)
y = df['Class']


# 80% train / 10% val / 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, test_size=0.20)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, random_state=42, test_size=0.50)


# fit scaler only on train to avoid data leakage
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_val[["Time", "Amount"]]   = scaler.transform(X_val[["Time", "Amount"]])
X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])


# SMOTE only on training data — never on val/test
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)


models = { 
    "rf": RandomForestClassifier(**rf_params),
    "logistic": LogisticRegressionCV(**lr_params),
    "Xgb": XGBClassifier(**xgb_params),
    "cat": CatBoostClassifier(**cat_params),
    "Lgbm": LGBMClassifier(**lgbm_params)
}


for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)


#Val block
# PR-AUC is primary metric — more reliable than ROC-AUC on imbalanced data
val_results = {}
for name, model in models.items():
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    val_results[name] = {
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall":    recall_score(y_val, y_pred),
        "F1":        f1_score(y_val, y_pred),
        "ROC-AUC":   roc_auc_score(y_val, y_proba),
        "PR-AUC":    average_precision_score(y_val, y_proba)
    }

val_df = pd.DataFrame(val_results).T
val_sorted = val_df.sort_values("PR-AUC", ascending=False)

print("\n" + "="*50)
print("VALIDATION RESULTS (used to select winner)")
print("="*50)
print(val_sorted)

# only consider models with recall >= 0.75 — high PR-AUC with poor recall isn't useful here
qualified = val_sorted[val_sorted['Recall'] >= 0.75]

if len(qualified) == 0:
    print("⚠️ No model met recall threshold, picking best PR-AUC anyway")
    qualified = val_sorted

winner_name = qualified.sort_values('PR-AUC', ascending=False).index[0]
winner_row  = qualified.loc[winner_name]

print(f"\nWINNER: {winner_name}")
print_metrics(winner_row)

# Test block
test_results = {}
for name, model in models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_results[name] = {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "PR-AUC":    average_precision_score(y_test, y_proba)
    }

test_df = pd.DataFrame(test_results).T

print("\n" + "="*50)
print("TEST RESULTS (final, unbiased — all models reported for transparency)")
print("="*50)
print(test_df)

# winner was selected on val above — test is reporting only, never re-selection
print(f"\nWINNER'S TEST PERFORMANCE: {winner_name}")
print_metrics(test_df.loc[winner_name])

joblib.dump(models[winner_name], "model.pkl")
joblib.dump(scaler, "scaler.pkl")
