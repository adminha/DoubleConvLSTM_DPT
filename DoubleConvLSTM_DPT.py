"""
Double Conv1D–LSTM with Differential Propagation Thresholding (DPT)

"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    classification_report,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    csv_path: str
    alt_name: str  # e.g. "ETH"
    lookback: int = 30  # timesteps (must be > 1 for temporal modeling)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 128
    epochs: int = 2000
    learning_rate: float = 4e-4
    early_stopping_patience: int = 50
    seed: int = 42
    out_dir: str = "results"


# -----------------------------
# Data utilities
# -----------------------------
def load_and_label(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    # Drop common timestamp/date columns (if present)
    for col in ["timestamp", "date", "Date", "time", "Time"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    alt_open = f"{cfg.alt_name}USDT_Open"
    alt_close = f"{cfg.alt_name}USDT_Close"
    label_col = f"{cfg.alt_name}_Candle_Color"

    if alt_open not in df.columns or alt_close not in df.columns:
        raise ValueError(
            f"Expected columns not found. Missing one of: {alt_open}, {alt_close}."
        )

    # Label: 1 if next-day candle is green (close - open > 0), else 0
    df[label_col] = np.where((df[alt_close] - df[alt_open]) > 0, 1, 0)

    # Shift label to represent next-day direction; drop last row (unknown future)
    df[label_col] = df[label_col].shift(-1)
    df = df.dropna().reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)

    return df


def chronological_split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    train_end = int(np.floor(train_ratio * n))
    val_end = int(np.floor((train_ratio + val_ratio) * n))
    if not (0 < train_end < val_end < n):
        raise ValueError("Invalid split sizes. Check ratios and dataset length.")
    return train_end, val_end


def fit_scaler_on_train(
    df: pd.DataFrame, feature_cols: list[str], train_end: int
) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(df.loc[: train_end - 1, feature_cols].values)
    return scaler


def apply_dpt(scaled_features: pd.DataFrame) -> pd.DataFrame:
    """
    Differential Propagation Thresholding:
    For each feature, encode whether it increased relative to the previous day.
    Uses only past information (diff), first row is set to 0.
    """
    dpt = (scaled_features.diff() > 0).astype(int)
    dpt = dpt.fillna(0).astype(int)
    return dpt


def build_sequences(
    X_df: pd.DataFrame, y: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping sequences:
    X[t] contains the last `lookback` rows ending at time t (inclusive).
    y[t] is the label at time t (already next-day direction from load_and_label).
    """
    X = X_df.values.astype(np.float32)
    y = y.astype(np.int32)

    if lookback < 2:
        raise ValueError("lookback must be >= 2 to model temporal dependencies.")

    Xs, ys = [], []
    for t in range(lookback - 1, len(X)):
        start = t - (lookback - 1)
        Xs.append(X[start : t + 1])
        ys.append(y[t])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


# -----------------------------
# Model
# -----------------------------
def build_model(input_shape: Tuple[int, int], lr: float) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu"),
            LSTM(128, return_sequences=True),
            Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu"),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu", kernel_initializer="he_uniform"),
            Dense(8, activation="relu", kernel_initializer="he_uniform"),
            Dense(1, activation="sigmoid"),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# Training + evaluation
# -----------------------------
def train_and_evaluate(cfg: Config) -> None:
    set_seed(cfg.seed)

    df = load_and_label(cfg)
    label_col = f"{cfg.alt_name}_Candle_Color"
    feature_cols = [c for c in df.columns if c != label_col]

    n = len(df)
    train_end, val_end = chronological_split_indices(n, cfg.train_ratio, cfg.val_ratio)

    # Fit scaler on TRAIN only (features only), then transform ALL using train-fitted scaler
    scaler = fit_scaler_on_train(df, feature_cols, train_end)
    scaled_all = pd.DataFrame(
        scaler.transform(df[feature_cols].values),
        columns=feature_cols,
    )

    # DPT on ALL feature columns (label excluded)
    dpt_all = apply_dpt(scaled_all)

    # Build sequences
    X_seq, y_seq = build_sequences(dpt_all, df[label_col].values, cfg.lookback)

    # Convert original split points to sequence-space indices
    offset = cfg.lookback - 1
    seq_train_end = train_end - offset
    seq_val_end = val_end - offset

    if seq_train_end <= 0 or seq_val_end <= seq_train_end or seq_val_end >= len(X_seq):
        raise ValueError(
            "Split points too early for the chosen lookback. "
            "Reduce lookback or use a longer dataset."
        )

    X_train, y_train = X_seq[:seq_train_end], y_seq[:seq_train_end]
    X_val, y_val = X_seq[seq_train_end:seq_val_end], y_seq[seq_train_end:seq_val_end]
    X_test, y_test = X_seq[seq_val_end:], y_seq[seq_val_end:]

    # Build model
    model = build_model(input_shape=(cfg.lookback, len(feature_cols)), lr=cfg.learning_rate)

    # Train (NO SHUFFLING)
    es = EarlyStopping(
        monitor="val_loss",
        patience=cfg.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=[es],
        verbose=1,
    )

    # Predict
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n=== Test metrics (threshold=0.5) ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"Macro Precision:   {prec_m:.4f}")
    print(f"Macro Recall:      {rec_m:.4f}")
    print(f"Macro F1-score:    {f1_m:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Output directory
    out_dir = Path(cfg.out_dir) / cfg.alt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save feature schema
    schema_path = out_dir / "feature_schema.txt"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("Feature columns (DPT-applied):\n")
        for c in feature_cols:
            f.write(f"- {c}\n")
        f.write(f"\nTotal features: {len(feature_cols)}\n")

    # Confusion matrix (labels: 0=Red, 1=Green)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Red (0)", "Green (1)"],
        yticklabels=["Red (0)", "Green (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({cfg.alt_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Training curves
    plt.figure()
    plt.plot(history.history.get("accuracy", []))
    plt.plot(history.history.get("val_accuracy", []))
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=300)
    plt.close()

    # Save metrics summary
    metrics_path = out_dir / "metrics_summary.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Test metrics (threshold=0.5)\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Balanced Accuracy: {bacc:.6f}\n")
        f.write(f"Macro Precision: {prec_m:.6f}\n")
        f.write(f"Macro Recall: {rec_m:.6f}\n")
        f.write(f"Macro F1-score: {f1_m:.6f}\n")
        f.write(f"MCC: {mcc:.6f}\n")

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Example usage (edit csv_path and alt_name):
    cfg = Config(
        csv_path="BTC-ETH_Merged_Data.csv",
        alt_name="ETH",
        lookback=30,
    )
    train_and_evaluate(cfg)
