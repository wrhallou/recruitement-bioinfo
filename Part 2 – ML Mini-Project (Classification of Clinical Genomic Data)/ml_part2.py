# core.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, balanced_accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# === CONFIG ===
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 0) Load ----
print(" ---- 0) Load ----")
df = pd.read_csv(
    "/data2/wassim_r/OutputData/test_dir/miniPrj/bioinfo/ml_dataset/ml_dataset_test.csv",
    sep=";"
)
target_col = "diagnosis"

numeric_cols = ["age", "prs", "feat1"] 
categorical_cols = ["subject_id", "gender", "is_smoker"]
snp_categorical = [f"var{i}" for i in range(1, 11)]

X = df.drop(columns=[target_col])
y = df[target_col]

# ---- 1) Exploratory Data Analysis (EDA) ----
print(" ---- 1) Exploratory Data Analysis ----")

plt.figure()
sns.countplot(x=target_col, data=df, palette="Set2")
plt.title("Target class distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.pdf"))
plt.close()

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{col}.pdf"))
    plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation between numeric features")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.pdf"))
plt.close()

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=target_col, y=col, data=df, palette="Set3")
    plt.title(f"{col} by {target_col}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"box_{col}_by_target.pdf"))
    plt.close()

for col in categorical_cols:
    plt.figure()
    sns.countplot(x=col, data=df, palette="Set2")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, f"count_{col}.pdf"))
    plt.close()

# ---- 2) Split ----
print(" ---- 2) Split ----")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---- 3) Preprocess ----
numeric_pre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])

categorical_pre = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

pre = ColumnTransformer(
    transformers=[
        ("num", numeric_pre, numeric_cols),
        ("cat", categorical_pre, categorical_cols + snp_categorical),
    ],
    remainder="drop"
)

# ---- 4a) Model A: Elastic-net multinomial logistic ----
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="saga",
    max_iter=5000,
    class_weight="balanced",
    penalty="elasticnet",
    l1_ratio=0.5
)

pipe_log = Pipeline([
    ("pre", pre),
    ("clf", logreg)
])

param_grid_log = {
    "clf__C": np.logspace(-3, 2, 10),
    "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_log = GridSearchCV(pipe_log, param_grid_log, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
grid_log.fit(X_train, y_train)

cal_log = CalibratedClassifierCV(grid_log.best_estimator_, method="isotonic", cv=3)
cal_log.fit(X_train, y_train)

# ---- 4b) Model B: HistGradientBoosting ----
hgb = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, random_state=42)

pipe_hgb = Pipeline([
    ("pre", pre),
    ("clf", hgb)
])

param_grid_hgb = {
    "clf__max_depth": [None, 3, 5, 7],
    "clf__learning_rate": [0.02, 0.05, 0.1],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__l2_regularization": [0.0, 0.1, 1.0]
}

grid_hgb = GridSearchCV(pipe_hgb, param_grid_hgb, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
grid_hgb.fit(X_train, y_train)

# ---- 5) Evaluation helpers ----
def eval_model(model, Xtr, Ytr, Xte, Yte, name="model"):
    pred = model.predict(Xte)
    print(f"\n=== {name} ===")
    print("Balanced accuracy:", balanced_accuracy_score(Yte, pred))
    print("Macro F1:", f1_score(Yte, pred, average="macro"))
    print("Per-class report:\n", classification_report(Yte, pred, digits=3))

    cm = confusion_matrix(Yte, pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_{name}.pdf"))
    plt.close()

    try:
        RocCurveDisplay.from_estimator(model, Xte, Yte)
        plt.title(f"ROC Curves - {name}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{name}.pdf"))
        plt.close()
    except Exception as e:
        print(f"ROC not available for {name}: {e}")

# ---- 6) Final test evaluation ----
print("---- 6) Final test evaluation ----")

models = {
    "ElasticNet_Calibrated": cal_log,
    "HistGB": grid_hgb.best_estimator_
}

for name, model in models.items():
    eval_model(model, X_train, y_train, X_test, y_test, name=name)

# Feature importance for HistGB
best_hgb = grid_hgb.best_estimator_.named_steps["clf"]
feature_names = grid_hgb.best_estimator_.named_steps["pre"].get_feature_names_out()
importances = best_hgb.feature_importances_
idx = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(8,6))
sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], palette="viridis")
plt.title("Top 15 Feature Importances - HistGB")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances_HistGB.pdf"))
plt.close()

print(f"\nAll plots and results saved in: {OUTPUT_DIR}")

