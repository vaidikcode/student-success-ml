"""
Neural networks with scikit-learn: MLPClassifier (pass/fail) and MLPRegressor (final score).
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .eda import _save

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SEED = 42


def split_and_scale(X, y, stratify=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if stratify else None)
    scaler = StandardScaler().fit(X_train)          # fit on TRAIN only -> no data leakage
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, scaler


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ----------------------------------------------------------------------------
# 1. Training curves: train vs validation loss & accuracy, epoch by epoch
# ----------------------------------------------------------------------------
def train_with_curves(X_train, y_train, epochs=150, **mlp_kwargs):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    clf = MLPClassifier(random_state=SEED, **mlp_kwargs)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        clf.partial_fit(X_tr, y_tr, classes=np.array([0, 1]))
        for split, (Xs, ys) in {"train": (X_tr, y_tr), "val": (X_val, y_val)}.items():
            hist[f"{split}_loss"].append(log_loss(ys, clf.predict_proba(Xs)))
            hist[f"{split}_acc"].append(clf.score(Xs, ys))
    return clf, hist


def plot_curves(histories: dict, name: str, title: str) -> None:
    fig, axes = plt.subplots(len(histories), 2, figsize=(13, 3.8 * len(histories)), squeeze=False)
    for row, (label, h) in enumerate(histories.items()):
        for col, metric in enumerate(["loss", "acc"]):
            ax = axes[row, col]
            ax.plot(h[f"train_{metric}"], label="train", color="#264653", linewidth=2)
            ax.plot(h[f"val_{metric}"], label="validation", color="#e76f51", linewidth=2)
            ax.set(xlabel="epoch", ylabel=metric, title=f"{label} — {'loss' if metric == 'loss' else 'accuracy'}")
            if metric == "acc":
                ax.set_ylim(0.3, 1.02)
            ax.legend()
        best = int(np.argmin(h["val_loss"]))
        axes[row, 0].axvline(best, color="gray", linestyle=":", label="best val epoch")
    fig.suptitle(title, fontsize=14, weight="bold")
    _save(fig, name)


def overfitting_experiment(X_train, y_train) -> None:
    """Same data, three models: underfit, overfit, and just right."""
    print("\n=== Overfitting vs underfitting ===")
    configs = {
        "UNDERFIT: 1 neuron + alpha=100": dict(hidden_layer_sizes=(1,), learning_rate_init=0.001, alpha=100.0),
        "OVERFIT: (256,128,64), no regularization": dict(hidden_layer_sizes=(256, 128, 64), learning_rate_init=0.005, alpha=0.0),
        "GOOD FIT: (32,16) + alpha=1.0": dict(hidden_layer_sizes=(32, 16), learning_rate_init=0.001, alpha=1.0),
    }
    histories = {}
    for label, cfg in configs.items():
        _, h = train_with_curves(X_train, y_train, epochs=120, **cfg)
        histories[label] = h
        print(f"  {label:<45} train acc {h['train_acc'][-1]:.3f} | val acc {h['val_acc'][-1]:.3f} "
              f"| gap {h['train_acc'][-1] - h['val_acc'][-1]:+.3f}")
    plot_curves(histories, "08_overfitting_vs_underfitting.png", "Underfitting vs overfitting vs a good fit")


# ----------------------------------------------------------------------------
# 2. Hyperparameter comparison
# ----------------------------------------------------------------------------
def hyperparameter_search(X_train, y_train) -> pd.DataFrame:
    print("\n=== Hyperparameter comparison (validation accuracy) ===")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    rows = []
    experiments = (
        [("activation", a, dict(activation=a)) for a in ["relu", "tanh", "logistic"]]
        + [("solver", s, dict(solver=s, learning_rate_init=0.01 if s == "sgd" else 0.001)) for s in ["adam", "sgd"]]
        + [("hidden_layer_sizes", str(h), dict(hidden_layer_sizes=h)) for h in [(8,), (32,), (64, 32), (128, 64, 32)]]
        + [("alpha", a, dict(alpha=a)) for a in [1e-5, 1e-2, 1.0, 10.0]]
        + [("learning_rate_init", lr, dict(learning_rate_init=lr)) for lr in [1e-4, 1e-3, 1e-2, 1e-1]]
    )
    base = dict(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=SEED)
    for group, value, override in experiments:
        clf = MLPClassifier(**{**base, **override}).fit(X_tr, y_tr)
        rows.append({"hyperparameter": group, "value": str(value), "val_accuracy": clf.score(X_val, y_val),
                     "epochs_run": clf.n_iter_})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
    for ax, (group, sub) in zip(axes, res.groupby("hyperparameter", sort=False)):
        bars = ax.bar(sub["value"], sub["val_accuracy"], color=sns.color_palette("mako", len(sub)))
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_title(group)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("validation accuracy")
    axes[0].set_ylim(0.3, 1.0)
    fig.suptitle("How each hyperparameter changes the MLP", fontsize=14, weight="bold")
    _save(fig, "09_hyperparameters.png")
    return res


def scaler_comparison(X, y) -> None:
    """Why scaling matters: raw vs MinMax vs Standard."""
    print("\n=== Effect of feature scaling ===")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    for name, scaler in [("no scaling", None), ("MinMaxScaler", MinMaxScaler()), ("StandardScaler", StandardScaler())]:
        a, b = (X_tr, X_te) if scaler is None else (scaler.fit_transform(X_tr), scaler.transform(X_te))
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=SEED).fit(a, y_tr)
        print(f"  {name:<16} test accuracy = {clf.score(b, y_te):.3f}")


# ----------------------------------------------------------------------------
# 3. Final classifier + evaluation plots
# ----------------------------------------------------------------------------
def evaluate_classifier(clf, X_test, y_test, numpy_mlp=None, X_test_np=None) -> dict:
    y_pred, y_proba = clf.predict(X_test), clf.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_proba)
    print("\n=== Final MLPClassifier on the held-out TEST set ===")
    print(classification_report(y_test, y_pred, target_names=["Failed", "Passed"]))
    print("  ROC-AUC:", round(metrics["roc_auc"], 4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(y_test, y_pred)
    labels = np.array([[f"{v}\n({v / cm.sum():.1%})" for v in row] for row in cm])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[0], annot_kws={"size": 13},
                xticklabels=["Failed", "Passed"], yticklabels=["Failed", "Passed"])
    axes[0].set(xlabel="predicted", ylabel="actual", title="Confusion matrix")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, linewidth=2.5, color="#2a9d8f", label=f"sklearn MLP (AUC = {metrics['roc_auc']:.3f})")
    if numpy_mlp is not None:
        p = numpy_mlp.predict_proba(X_test_np)
        f2, t2, _ = roc_curve(y_test, p)
        axes[1].plot(f2, t2, linewidth=2, linestyle="--", color="#e9c46a",
                     label=f"from-scratch NumPy MLP (AUC = {roc_auc_score(y_test, p):.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle=":", label="random guess")
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    axes[1].legend(loc="lower right")
    _save(fig, "10_confusion_matrix_roc.png")
    return metrics


def plot_decision_boundary(df) -> None:
    """Train a 2-feature MLP so we can SEE what it learned."""
    feats = ["study_hours_week", "sleep_hours"]
    X, y = df[feats].to_numpy(), df["passed"].to_numpy()
    scaler = StandardScaler().fit(X)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, act in zip(axes, ["relu", "tanh", "logistic"]):
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation=act, max_iter=2000, random_state=SEED,
                            alpha=0.01).fit(scaler.transform(X), y)
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                             np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
        grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        zz = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
        cf = ax.contourf(xx, yy, zz, levels=20, cmap="RdYlGn", alpha=0.75)
        ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlGn", edgecolor="k", s=12, linewidth=0.3)
        ax.set(xlabel="study hours / week", ylabel="sleep hours", title=f"activation = '{act}'")
    fig.colorbar(cf, ax=axes, label="P(pass)", shrink=0.85)
    fig.suptitle("Decision boundary: the network found the 'sleep sweet spot' by itself",
                 fontsize=14, weight="bold", y=1.02)
    fig.savefig(_fig_path("11_decision_boundary.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[eda] saved figures/11_decision_boundary.png")


def _fig_path(name):
    from .eda import FIG_DIR
    FIG_DIR.mkdir(exist_ok=True)
    return FIG_DIR / name


# ----------------------------------------------------------------------------
# 4. Regression: predict the exact final score
# ----------------------------------------------------------------------------
def run_regression(X, y, linear_baseline=None) -> dict:
    print("\n=== MLPRegressor: predicting the exact final score ===")
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y, stratify=False)
    reg = MLPRegressor(hidden_layer_sizes=(32,), alpha=10.0, learning_rate_init=0.001, max_iter=3000,
                       early_stopping=True, validation_fraction=0.15, n_iter_no_change=50, random_state=SEED)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    metrics = {"mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2_score(y_test, pred), "epochs": reg.n_iter_}
    print(f"  MSE = {mse:.2f} | RMSE = {metrics['rmse']:.2f} points | R² = {metrics['r2']:.3f} "
          f"| early stopping after {reg.n_iter_} epochs")
    if linear_baseline:
        print(f"  (NumPy normal-equation linear regression baseline: RMSE = {linear_baseline['rmse']:.2f}, "
              f"R² = {linear_baseline['r2']:.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    axes[0].plot(reg.loss_curve_, label="training loss", color="#264653", linewidth=2)
    ax2 = axes[0].twinx()
    ax2.plot(reg.validation_scores_, label="validation R²", color="#e76f51", linewidth=2)
    axes[0].set(xlabel="epoch", ylabel="loss (squared error / 2)", title="Training loss & validation R²")
    ax2.set_ylabel("validation R²")
    axes[0].legend(loc="center right")
    ax2.legend(loc="lower right")

    axes[1].scatter(y_test, pred, alpha=0.5, s=15, color="#2a9d8f")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axes[1].plot(lims, lims, "r--", label="perfect prediction")
    axes[1].set(xlabel="actual score", ylabel="predicted score", title=f"Predicted vs actual (R² = {metrics['r2']:.3f})")
    axes[1].legend()

    sns.histplot(y_test - pred, kde=True, ax=axes[2], color="#8e7dbe")
    axes[2].set(xlabel="residual (actual − predicted)", title=f"Residuals (RMSE = {metrics['rmse']:.2f})")
    _save(fig, "12_regression.png")
    return metrics


def save_model(clf, scaler, features, metrics) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "features": features}, RESULTS_DIR / "pass_predictor.joblib")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print("\n[models] saved results/pass_predictor.joblib and results/metrics.json")
