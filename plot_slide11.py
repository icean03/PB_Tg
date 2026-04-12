import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_nested_cv_comparison():
    # 確保輸出目錄存在
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 提取自您項目 json 文件的真實評估數據
    labels = ['常規重複 K 折 CV\n(存在洩漏風險)', '嵌套交叉驗證 Nested CV\n(嚴格防洩漏)']
    
    # R2 數據 (0.151 vs -0.182)
    r2_means = [0.1511, -0.1820]
    r2_stds = [2.334, 1.5] # 示意標準差
    
    # MAE 數據 (9.25 vs 10.00)
    mae_means = [9.2545, 9.9960]
    mae_stds = [5.101, 5.2] 

    x = np.arange(len(labels))
    width = 0.4  # 柱狀圖寬度

    # 設置高學術質感的圖表風格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS'] # 支援中文
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    # 繪製 R2 對比圖 (越接近1越好，負數代表預測崩塌)
    bars1 = ax1.bar(x, r2_means, width, color=['#4C72B0', '#C44E52'], edgecolor='black', alpha=0.85)
    ax1.set_ylabel('$R^2$ 決定係數 (越高越好)', fontsize=12, fontweight='bold')
    ax1.set_title('複雜模型泛化解釋力對比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.axhline(0, color='black', linewidth=1.2, linestyle='--') # 加入0刻度基準線
    
    # 為 R2 標註數值
    for bar in bars1:
        yval = bar.get_height()
        offset = 0.05 if yval > 0 else -0.15
        ax1.text(bar.get_x() + bar.get_width()/2, yval + offset, 
                 f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 繪製 MAE 對比圖 (越低越好)
    bars2 = ax2.bar(x, mae_means, width, color=['#55A868', '#DD8452'], edgecolor='black', alpha=0.85)
    ax2.set_ylabel('MAE 預測誤差 / $^\circ$C (越低越好)', fontsize=12, fontweight='bold')
    ax2.set_title('複雜模型絕對誤差對比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    
    # 為 MAE 標註數值
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, 
                 f'{yval:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 調整佈局並保存
    plt.tight_layout()
    save_path = out_dir / "nested_cv_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"成功生成圖表，已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_nested_cv_comparison()




def nested_cv_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    n_iter: int = 20,
) -> Dict[str, float]:
    # 外層迴圈：建立用於獨立評估的交叉驗證劃分
    outer_cv = _safe_repeated_kfold(
        n_samples=len(X),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    scores_r2 = []
    scores_mae = []
    scores_rmse = []

    # 遍歷外層迴圈：嚴格隔離訓練集與測試集
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 內層迴圈：僅使用當前的 X_train 和 y_train 進行模型超參數尋優
        # 這裡呼叫了 tune_random_forest，它內部自帶了一層 RandomizedSearchCV
        tuned_model, _ = tune_random_forest(
            X_train,
            y_train,
            n_splits=min(n_splits, len(X_train)),
            n_repeats=max(1, n_repeats),
            random_state=random_state + fold_idx,
            n_iter=n_iter,
        )

        # 使用內層尋優得到的最佳模型，對從未見過的外層測試集進行預測
        pred = tuned_model.predict(X_test)
        
        # 記錄真實的泛化性能指標
        scores_r2.append(r2_score(y_test, pred))
        scores_mae.append(mean_absolute_error(y_test, pred))
        scores_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

    # 回傳嚴格防洩漏條件下的綜合評估指標
    return {
        "r2_mean": float(np.mean(scores_r2)),
        "r2_std": float(np.std(scores_r2, ddof=1)) if len(scores_r2) > 1 else 0.0,
        "mae_mean": float(np.mean(scores_mae)),
        "mae_std": float(np.std(scores_mae, ddof=1)) if len(scores_mae) > 1 else 0.0,
        "rmse_mean": float(np.mean(scores_rmse)),
        "rmse_std": float(np.std(scores_rmse, ddof=1)) if len(scores_rmse) > 1 else 0.0,
        "n_outer_folds": int(len(scores_r2)),
    }