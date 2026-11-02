import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "water_quality_10000_balanced.csv"
)

df = pd.read_csv(DATASET_PATH)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["class"].value_counts())


# ============================================================
# 3. CHECK MISSING VALUES AND WATER QUALITY RULES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing rows if any
df = df.dropna().reset_index(drop=True)

print("\nWater quality classification rules:")
print("  Normal   : pH Value < 7 AND Turb < 2500 AND Temperature < 29.06 °C")
print("  Abnormal : at least one normal condition is violated")


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

FEATURES = [
    "PH",
    "Trub",
    "Temperature"
]

TARGET = "class"

# Verify that the generated dataset contains the expected columns.
required_columns = FEATURES + [TARGET]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}. "
        f"Available columns: {df.columns.tolist()}"
    )

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Ensure sensor features are numeric.
for col in FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove rows that became invalid after numeric conversion.
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# Normalize class text so the target is consistent.
y = y.astype(str).str.strip().str.title()

# Validate the target against the rule used to generate the dataset.
rule_normal = (
    (X["PH"] < 7) &
    (X["Trub"] < 2500) &
    (X["Temperature"] < 29.06)
)

rule_class = np.where(rule_normal, "Normal", "Abnormal")

mismatch_count = np.sum(y.to_numpy() != rule_class)

print(f"\nRule validation mismatches: {mismatch_count}")

if mismatch_count > 0:
    print("WARNING: Some rows do not match the supplied water-quality classification rule.")

# ============================================================
# 5. ENCODE TARGET CLASSES
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nWater Quality Class Encoding:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i} -> {class_name}")


# Save label encoder
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nScaler saved as:")
print("models/scaler.pkl")


# ============================================================
# 8. DEFINE ML MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=None
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        C=10,
        probability=True,
        random_state=42
    )
}


# ============================================================
# 9. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING AND COMPARISON")
print("=" * 70)

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Tree-based models don't require scaling,
    # but using scaled features consistently is acceptable here.
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = model

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# Save results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ============================================================
# 11. FIND BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_accuracy = results_df.iloc[0]["Accuracy"]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model :", best_model_name)
print("Accuracy   :", f"{best_accuracy * 100:.2f}%")


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

best_model_filename = (
    "models/best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_filename
)

print("\nBest model saved:")
print(best_model_filename)


# Also save model using its name
safe_name = best_model_name.lower().replace(" ", "_")

joblib.dump(
    best_model,
    f"models/{safe_name}.pkl"
)


# ============================================================
# 13. MODEL COMPARISON GRAPH - ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy (%)")
plt.title("ML Algorithm Accuracy Comparison")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "graphs/model_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. COMPARISON GRAPH - ALL METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

plot_df = results_df.copy()

for metric in metrics:
    plot_df[metric] = plot_df[metric] * 100

plot_df = plot_df.set_index("Model")

plot_df[metrics].plot(
    kind="bar",
    figsize=(12, 7)
)

plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Score (%)")
plt.title("ML Algorithm Performance Comparison")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 100)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    "graphs/model_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. CONFUSION MATRIX FOR BEST MODEL
# ============================================================

best_predictions = best_model.predict(
    X_test_scaled
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "graphs/best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    best_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print(report)

with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(
        f"Best Model: {best_model_name}\n"
    )

    file.write(
        f"Accuracy: {best_accuracy:.4f}\n\n"
    )

    file.write(report)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(feature_df.to_string(index=False))

    feature_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(
        f"Feature Importance - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        "graphs/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 18. SAVE DATASET INFORMATION
# ============================================================

dataset_info = {
    "Number of Samples": len(df),
    "Number of Features": len(FEATURES),
    "Features": ", ".join(FEATURES),
    "Number of Classes": len(label_encoder.classes_),
    "Classes": ", ".join(label_encoder.classes_),
    "Best Model": best_model_name,
    "Best Accuracy": best_accuracy
}

info_df = pd.DataFrame(
    [dataset_info]
)

info_df.to_csv(
    "results/dataset_and_model_info.csv",
    index=False
)


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print("\nMODELS/")
print("  best_model.pkl")
print("  scaler.pkl")
print("  label_encoder.pkl")

print("\nGRAPHS/")
print("  model_accuracy_comparison.png")
print("  model_metrics_comparison.png")
print("  best_model_confusion_matrix.png")
print("  feature_importance.png")

print("\nRESULTS/")
print("  model_comparison.csv")
print("  classification_report.txt")
print("  feature_importance.csv")
print("  dataset_and_model_info.csv")

print("\nBest Model:", best_model_name)
print(
    "Best Accuracy:",
    f"{best_accuracy * 100:.2f}%"
)