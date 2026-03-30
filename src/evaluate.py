from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    figures = base_dir / "figures"
    metrics = base_dir / "metrics"
    models = base_dir / "models"
    for p in (figures, metrics, models):
        p.mkdir(parents=True, exist_ok=True)
    return {"figures": figures, "metrics": metrics, "models": models}


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_pred_vs_actual(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=60, alpha=0.8)
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="y=x")
    plt.xlabel("实测 Tg (°C)")
    plt.ylabel("预测 Tg (°C)")
    plt.title("预测值 vs 实测值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    residual = y_true - y_pred
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residual, s=60, alpha=0.8)
    plt.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
    plt.xlabel("预测 Tg (°C)")
    plt.ylabel("残差 (实测-预测)")
    plt.title("残差图")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def permutation_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_repeats: int = 30,
) -> pd.DataFrame:
    model.fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    fi = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    return fi.reset_index(drop=True)


def plot_feature_importance(fi_df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(7, max(4, 0.5 * len(fi_df))))
    sns.barplot(data=fi_df, y="feature", x="importance_mean", orient="h")
    plt.xlabel("Permutation Importance (MAE decrease)")
    plt.ylabel("特征")
    plt.title("特征重要性")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
