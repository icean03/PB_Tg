from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold


@dataclass
class FeatureSelectionResult:
    selected_features: List[str]
    ranking: Dict[str, int]
    n_selected: int


def select_features_rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    max_cv_splits: int = 5,
    min_features_to_select: int = 2,
) -> FeatureSelectionResult:
    """在小样本场景下用 RFECV 做稳健特征筛选。

    若样本数/特征数过少，则直接返回全部特征，避免过拟合式筛选。
    """
    n_samples, n_features = X.shape
    if n_samples < 8 or n_features <= min_features_to_select:
        ranking = {col: 1 for col in X.columns}
        return FeatureSelectionResult(list(X.columns), ranking, len(X.columns))

    cv_splits = min(max_cv_splits, n_samples)
    cv_splits = max(3, cv_splits)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    estimator = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
        scoring="neg_mean_absolute_error",
        min_features_to_select=min(min_features_to_select, n_features),
        n_jobs=-1,
    )
    selector.fit(X_imp, y)

    selected_features = X.columns[selector.support_].tolist()
    ranking = {col: int(rank) for col, rank in zip(X.columns, selector.ranking_)}

    return FeatureSelectionResult(
        selected_features=selected_features,
        ranking=ranking,
        n_selected=len(selected_features),
    )
