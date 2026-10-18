import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

SEED = 42

def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"^.{0,60}\((reuters|ap|afp)\)\s*[-–—]?\s*", "", t, flags=re.I)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def main():
    dataset_path = "WELFake_Dataset.csv"
    model_output_dir = "backend/model"
    figs_output_dir = "frontend/static/figs"

    print("Step 1: Reading dataset...", flush=True)
    df = pd.read_csv(dataset_path, usecols=["title", "text", "label"])
    print(f"Loaded {len(df)} rows.", flush=True)

    print("Step 2: Preparing text and cleaning...", flush=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + ". " + df["text"]).str.strip()
    # In WELFake_Dataset.csv: label 1 is FAKE, 0 is REAL.
    # We map to standard: FAKE = 0, REAL = 1 (matching notebook y = 1 - label).
    df["y"] = 1 - df["label"].astype(int)

    df["content"] = df["content"].map(clean_text)
    before_len = len(df)
    df = df[df["content"].str.split().str.len() >= 10]
    df = df.drop_duplicates(subset="content")
    print(f"Filtered {before_len} -> {len(df)} rows (removed short text & duplicates).", flush=True)
    print("Class counts (0=FAKE, 1=REAL):", flush=True)
    print(df["y"].value_counts().to_dict(), flush=True)

    print("Step 3: Train/Test Split (80/20)...", flush=True)
    train_full, test_full = train_test_split(
        df[["content", "y"]],
        test_size=0.2,
        stratify=df["y"],
        random_state=SEED,
    )

    # 20k train and 5k test matches notebook TRAIN_N=20000, TEST_N=5000
    N_TRAIN = 20000
    N_TEST = 5000
    train_df = train_full.sample(min(N_TRAIN, len(train_full)), random_state=SEED).reset_index(drop=True)
    test_df = test_full.sample(min(N_TEST, len(test_full)), random_state=SEED).reset_index(drop=True)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}", flush=True)

    print("Step 4: Vectorizing with TF-IDF (max_features=50000, n-grams 1-2)...", flush=True)
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(train_df["content"])
    X_test = tfidf.transform(test_df["content"])
    y_train = train_df["y"].values
    y_test = test_df["y"].values

    print("Step 5: Fitting Calibrated LinearSVC (Linear Support Vector Machine)...", flush=True)
    base_svc = LinearSVC(random_state=SEED, max_iter=2000)
    clf = CalibratedClassifierCV(estimator=base_svc, cv=3)
    clf.fit(X_train, y_train)

    print("Step 6: Evaluating model...", flush=True)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60, flush=True)
    print("EVALUATION RESULTS (TF-IDF + LinearSVC on WELFake)", flush=True)
    print("=" * 60, flush=True)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print(f"Precision: {prec:.4f}", flush=True)
    print(f"Recall   : {rec:.4f}", flush=True)
    print(f"F1-Score : {f1:.4f}", flush=True)
    print(f"ROC-AUC  : {roc_auc:.4f}", flush=True)
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:", flush=True)
    print(cm, flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], digits=4), flush=True)
    print("=" * 60, flush=True)

    print("Step 7: Saving model pipeline...", flush=True)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(figs_output_dir, exist_ok=True)

    pipeline = {
        "vectorizer": tfidf,
        "model": clf,
        "classes": ["FAKE", "REAL"],
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
    }
    model_save_path = os.path.join(model_output_dir, "model_pipeline.joblib")
    joblib.dump(pipeline, model_save_path, compress=3)
    print(f"Model saved successfully to {model_save_path}!", flush=True)

    print("Step 8: Generating evaluation plots...", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
    disp.plot(ax=axs[0], cmap="Blues", colorbar=False)
    axs[0].set_title("Confusion Matrix (LinearSVC)", fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    axs[1].plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    axs[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axs[1].legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(figs_output_dir, "evaluation_metrics.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation plot to {plot_path}", flush=True)
    print("TRAINING AND EVALUATION COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
