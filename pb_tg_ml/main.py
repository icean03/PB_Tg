from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.data_preprocess import load_dataset, prepare_features
from src.evaluate import (
    ensure_output_dirs,
    permutation_feature_importance,
    plot_feature_importance,
    plot_pred_vs_actual,
    plot_residual,
    save_json,
)
from src.train_cv import (
    benchmark_models,
    cv_predictions_for_plot,
    evaluate_model_cv,
    nested_cv_random_forest,
    tune_random_forest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="聚丁二烯 Tg 预测（随机森林 + 小样本稳健策略）"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/pb_tg_demo.csv",
        help="输入 CSV 数据路径",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Tg_C",
        help="目标列列名",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="交叉验证折数",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="重复次数（RepeatedKFold）",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--search-iter",
        type=int,
        default=20,
        help="随机搜索迭代次数",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    out_dirs = ensure_output_dirs(Path(args.outdir))

    print("[1/7] 读取数据...")
    df = load_dataset(data_path)

    print("[2/7] 数据预处理与特征构建...")
    X, y, summary = prepare_features(df, target_col=args.target)
    print(
        f"样本数={summary.n_samples}, 特征数={summary.n_features}, 特征={summary.feature_columns}"
    )

    print("[3/7] 基线模型对比（重复K折）...")
    benchmark_df = benchmark_models(
        X,
        y,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    benchmark_path = out_dirs["metrics"] / "model_benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False, encoding="utf-8-sig")

    print("[4/7] 随机森林调参...")
    best_model, best_params = tune_random_forest(
        X,
        y,
        n_splits=args.n_splits,
        n_repeats=max(3, args.n_repeats // 2),
        random_state=args.random_state,
        n_iter=args.search_iter,
    )
    save_json(best_params, out_dirs["metrics"] / "best_params.json")

    print("[5/7] 最终模型重复K折评估...")
    m = evaluate_model_cv(
        best_model,
        X,
        y,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    final_metrics = {
        "r2_mean": m.r2_mean,
        "r2_std": m.r2_std,
        "mae_mean": m.mae_mean,
        "mae_std": m.mae_std,
        "rmse_mean": m.rmse_mean,
        "rmse_std": m.rmse_std,
        "used_features": X.columns.tolist(),
    }
    save_json(final_metrics, out_dirs["metrics"] / "random_forest_cv_metrics.json")

    print("[6/7] 嵌套CV评估（防信息泄漏）...")
    nested_metrics = nested_cv_random_forest(
        X,
        y,
        n_splits=args.n_splits,
        n_repeats=1,
        random_state=args.random_state,
        n_iter=min(8, args.search_iter),
    )
    save_json(nested_metrics, out_dirs["metrics"] / "random_forest_nested_cv_metrics.json")

    print("[7/7] 生成可视化与保存模型...")
    y_pred, cv_plot_metrics = cv_predictions_for_plot(
        best_model,
        X,
        y,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    plot_pred_vs_actual(y, y_pred, out_dirs["figures"] / "pred_vs_actual.png")
    plot_residual(y, y_pred, out_dirs["figures"] / "residual_plot.png")

    fi_df = permutation_feature_importance(
        best_model,
        X,
        y,
        random_state=args.random_state,
    )
    fi_df.to_csv(
        out_dirs["metrics"] / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_feature_importance(fi_df, out_dirs["figures"] / "feature_importance.png")

    model_payload = {
        "model": best_model.fit(X, y),
        "features": X.columns.tolist(),
        "target": args.target,
    }
    joblib.dump(model_payload, out_dirs["models"] / "random_forest_tg_model.joblib")

    print("\n===== 运行完成 =====")
    print("基线对比文件:", benchmark_path)
    print("最终CV指标:", out_dirs["metrics"] / "random_forest_cv_metrics.json")
    print("嵌套CV指标:", out_dirs["metrics"] / "random_forest_nested_cv_metrics.json")
    print("调参结果:", out_dirs["metrics"] / "best_params.json")
    print("可视化目录:", out_dirs["figures"])
    print("模型文件:", out_dirs["models"] / "random_forest_tg_model.joblib")
    print(
        "绘图CV单次指标（用于可视化）：",
        f"R2={cv_plot_metrics['r2']:.3f}, MAE={cv_plot_metrics['mae']:.3f}, RMSE={cv_plot_metrics['rmse']:.3f}",
    )


if __name__ == "__main__":
    main()
