from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class ModelMetrics:
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float


def _safe_repeated_kfold(
    n_samples: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> RepeatedKFold:
    if n_samples < 3:
        raise ValueError("样本数太少（<3），无法进行交叉验证。")
    splits = max(2, min(n_splits, n_samples))
    repeats = max(1, n_repeats)
    return RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=random_state)


def _metrics_from_cv_result(cv_result: Dict[str, np.ndarray]) -> ModelMetrics:
    r2 = cv_result["test_r2"]
    mae = -cv_result["test_neg_mae"]
    rmse = -cv_result["test_neg_rmse"]
    return ModelMetrics(
        r2_mean=float(np.mean(r2)),
        r2_std=float(np.std(r2, ddof=1)) if len(r2) > 1 else 0.0,
        mae_mean=float(np.mean(mae)),
        mae_std=float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
        rmse_mean=float(np.mean(rmse)),
        rmse_std=float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
    )


def benchmark_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    cv = _safe_repeated_kfold(len(X), n_splits, n_repeats, random_state)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }

    models = {
        "LinearRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "SVR": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(C=10.0, epsilon=0.5, kernel="rbf")),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    rows = []
    for name, model in models.items():
        cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        m = _metrics_from_cv_result(cv_result)
        rows.append(
            {
                "model": name,
                "r2_mean": m.r2_mean,
                "r2_std": m.r2_std,
                "mae_mean": m.mae_mean,
                "mae_std": m.mae_std,
                "rmse_mean": m.rmse_mean,
                "rmse_std": m.rmse_std,
            }
        )

    return pd.DataFrame(rows).sort_values(by="mae_mean", ascending=True).reset_index(drop=True)


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    n_iter: int = 20,
) -> Tuple[Pipeline, Dict[str, object]]:
    cv = _safe_repeated_kfold(len(X), n_splits, n_repeats, random_state)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "rf__n_estimators": [200, 300, 500, 800],
        "rf__max_depth": [None, 5, 8, 12, 20],
        "rf__min_samples_split": [2, 3, 4, 6, 8],
        "rf__min_samples_leaf": [1, 2, 3, 4],
        "rf__max_features": ["sqrt", 0.6, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=max(5, n_iter),
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def evaluate_model_cv(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> ModelMetrics:
    cv = _safe_repeated_kfold(len(X), n_splits, n_repeats, random_state)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return _metrics_from_cv_result(cv_result)


def cv_predictions_for_plot(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    splits = max(2, min(n_splits, len(X)))
    kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    pred = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    metrics = {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }
    return pred, metrics


def nested_cv_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    n_iter: int = 20,
) -> Dict[str, float]:
    outer_cv = _safe_repeated_kfold(
        n_samples=len(X),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    scores_r2 = []
    scores_mae = []
    scores_rmse = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        tuned_model, _ = tune_random_forest(
            X_train,
            y_train,
            n_splits=min(n_splits, len(X_train)),
            n_repeats=max(1, n_repeats),
            random_state=random_state + fold_idx,
            n_iter=n_iter,
        )

        pred = tuned_model.predict(X_test)
        scores_r2.append(r2_score(y_test, pred))
        scores_mae.append(mean_absolute_error(y_test, pred))
        scores_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

    return {
        "r2_mean": float(np.mean(scores_r2)),
        "r2_std": float(np.std(scores_r2, ddof=1)) if len(scores_r2) > 1 else 0.0,
        "mae_mean": float(np.mean(scores_mae)),
        "mae_std": float(np.std(scores_mae, ddof=1)) if len(scores_mae) > 1 else 0.0,
        "rmse_mean": float(np.mean(scores_rmse)),
        "rmse_std": float(np.std(scores_rmse, ddof=1)) if len(scores_rmse) > 1 else 0.0,
        "n_outer_folds": int(len(scores_r2)),
    }
