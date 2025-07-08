# BTC Price Direction Prediction - gridSearchBest

本项目为比特币价格涨跌预测的深度学习自动化实验平台，支持多模型、多参数大规模超参数搜索，适配多GPU环境，适合金融量化、学术研究和工业部署。

## 目录结构

- `auto_model_search.py`：主自动化超参数搜索脚本，支持全模型、全参数空间遍历，自动保存每组实验结果和全局最优。
- `unified_models.py`：集成所有主流时序深度模型（LSTM/GRU/Transformer/TCN/Ensemble/TabNet等）及通用工具函数。
- `results/`：自动保存每组实验结果和全局最优结果。
- `btcusdt.json`：主数据文件（需自行准备，格式见下）。
- `btcusdt_small.json`：小数据集测试样例（可选）。

## 主要功能

- 支持 LSTM、GRU、Transformer、TCN、CNN-LSTM、Ensemble、TabNet 等多种模型。
- 自动化超参数搜索，支持 learning_rate、hidden_size、dropout、attention_type 等主流参数。
- 多GPU自动适配，支持大批量训练。
- 自动保存每组实验结果和全局最优结果，便于后续分析。
- 训练/验证过程输出详细日志和多指标（acc、auc、f1等）。

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision scikit-learn pandas numpy
```

### 2. 准备数据

- 将你的BTC历史K线数据保存为 `btcusdt.json`，格式为每行一个dict，包含至少如下字段：
  - open_time, open, high, low, close, volume

### 3. 运行自动化实验

```bash
python auto_model_search.py
```

- 结果将自动保存在 `results/` 目录下，包括每组参数的结果和全局最优结果。

### 4. 小数据集测试

如需快速测试，可用 `btcusdt_small.json` 替换主数据文件。

## 扩展建议

- 可集成 Optuna、Ray Tune 等自动调参工具。
- 支持更多特征、外部数据、模型融合等。
- 结果可视化、自动报告生成等。

---

如有问题欢迎提 issue 或 PR！ 
