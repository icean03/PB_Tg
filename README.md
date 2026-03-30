# 基于机器学习的聚丁二烯玻璃化转变温度预测

本项目提供一个可直接运行的 Python 小样本稳健建模框架：

- 目标：用聚丁二烯结构/分子量参数预测 Tg（`Tg_C`）
- 主模型：随机森林回归（RandomForestRegressor）
- 小样本策略：特征筛选（RFECV）+ 重复 K 折交叉验证（RepeatedKFold）
- 输出：模型指标、最佳参数、特征重要性、可视化图、模型文件

---

## 1. 目录结构

```text
pb_tg_ml/
├─ data/
│  └─ raw/
│     ├─ pb_tg_demo.csv         # 演示数据（可直接跑通）
│     └─ pb_tg_template.csv     # 你的数据模板
├─ outputs/
│  ├─ figures/                  # 图像输出目录（运行后生成）
│  ├─ metrics/                  # 指标输出目录（运行后生成）
│  └─ models/                   # 模型输出目录（运行后生成）
├─ src/
│  ├─ data_preprocess.py
│  ├─ feature_selection.py
│  ├─ train_cv.py
│  └─ evaluate.py
├─ main.py
└─ requirements.txt
```

---

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 3. 数据字段说明

至少建议包含以下列（列名尽量一致）：

- `cis_1_4_pct`
- `trans_1_4_pct`
- `vinyl_1_2_pct`（可选）
- `Mn`
- `Mw`
- `PDI`（可选，若缺失会由 `Mw/Mn` 自动计算）
- `Tg_C`（目标列）

可附加：`sample_id`, `ref_id`, `material_type`, `test_method`, `notes` 等元信息。

---

## 4. 运行方式

### 4.1 直接用演示数据跑通

```bash
python main.py
```

### 4.2 使用你自己的数据

```bash
python main.py --data data/raw/你的数据.csv --target Tg_C
```

可调参数示例：

```bash
python main.py --data data/raw/你的数据.csv --n-splits 5 --n-repeats 10 --random-state 42
```

---

## 5. 输出结果

运行后会生成：

- `outputs/metrics/model_benchmark.csv`：线性回归/岭回归/SVR/RF 对比（重复 K 折均值±标准差）
- `outputs/metrics/random_forest_cv_metrics.json`：最终随机森林指标
- `outputs/metrics/best_params.json`：最佳超参数
- `outputs/metrics/feature_importance.csv`：置换重要性
- `outputs/figures/pred_vs_actual.png`：预测值-实测值散点图
- `outputs/figures/residual_plot.png`：残差图
- `outputs/figures/feature_importance.png`：特征重要性图
- `outputs/models/random_forest_tg_model.joblib`：训练好的模型（含特征名）

---

## 6. 小样本建议

1. 优先保证数据质量（单位统一、缺失处理、可追溯来源）  
2. 先控制特征数量，避免过拟合  
3. 报告交叉验证的“均值 ± 标准差”，不要只报单个分数  
4. 论文中配合材料机理解释（顺反结构、分子量分布对链段运动与 Tg 的影响）
