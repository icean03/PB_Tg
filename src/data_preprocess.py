from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


DEFAULT_FEATURE_CANDIDATES = [
    "cis_1_4_pct",
    "trans_1_4_pct",
    "vinyl_1_2_pct",
    "Mn",
    "Mw",
    "PDI",
]


@dataclass
class DataSummary:
    n_samples: int
    n_features: int
    feature_columns: List[str]
    dropped_for_missing_target: int


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_pdi(df: pd.DataFrame) -> pd.DataFrame:
    has_mn = "Mn" in df.columns
    has_mw = "Mw" in df.columns
    has_pdi = "PDI" in df.columns

    if not has_mn or not has_mw:
        return df

    df["Mn"] = pd.to_numeric(df["Mn"], errors="coerce")
    df["Mw"] = pd.to_numeric(df["Mw"], errors="coerce")

    if not has_pdi:
        df["PDI"] = df["Mw"] / df["Mn"]
        return df

    df["PDI"] = pd.to_numeric(df["PDI"], errors="coerce")
    missing_mask = df["PDI"].isna() & df["Mn"].notna() & df["Mw"].notna()
    df.loc[missing_mask, "PDI"] = df.loc[missing_mask, "Mw"] / df.loc[missing_mask, "Mn"]
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Tg_C",
    feature_candidates: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, DataSummary]:
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在，请检查数据列名。")

    work_df = df.copy()
    work_df = infer_pdi(work_df)

    feature_candidates = feature_candidates or DEFAULT_FEATURE_CANDIDATES
    existing_candidates = [c for c in feature_candidates if c in work_df.columns]

    if not existing_candidates:
        numeric_cols = work_df.select_dtypes(include="number").columns.tolist()
        existing_candidates = [c for c in numeric_cols if c != target_col]

    if not existing_candidates:
        raise ValueError("未找到可用特征列，请检查数据是否包含数值型输入参数。")

    work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
    dropped_for_missing_target = int(work_df[target_col].isna().sum())
    work_df = work_df.dropna(subset=[target_col]).reset_index(drop=True)

    for col in existing_candidates:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    usable_features = []
    for col in existing_candidates:
        missing_ratio = work_df[col].isna().mean()
        unique_count = work_df[col].nunique(dropna=True)
        if missing_ratio <= 0.5 and unique_count > 1:
            usable_features.append(col)

    if not usable_features:
        raise ValueError("候选特征在当前数据中不可用（缺失过多或无变化）。")

    X = work_df[usable_features].copy()
    y = work_df[target_col].copy()

    summary = DataSummary(
        n_samples=len(work_df),
        n_features=len(usable_features),
        feature_columns=usable_features,
        dropped_for_missing_target=dropped_for_missing_target,
    )
    return X, y, summary
