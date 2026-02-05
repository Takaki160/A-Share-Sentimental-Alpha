
---

# A-Share-Sentimental-Alpha: 融合深度语义特征的 A 股截面收益率预测系统

## 项目定位

本项目构建了一套处理 A 股年报非结构化数据的量化策略流水线。系统通过提取**管理层讨论与分析（MD&A）中的深层语义特征，结合传统基本面因子，利用 Learning to Rank (LTR) 框架预测个股在财报披露后的行业中性化超额收益（Alpha）**。

---

## 核心特性

### 1. 消除预见偏差

* **公告日对齐**：严格基于每家上市公司的 **实际披露日** 触发信号。
* **时序净化验证**：采用滚动窗口训练模式，并在训练集与测试集之间加入 **60 天缓冲区**，确保无数据交叉污染。

### 2. 特征蒸馏与降噪

* **语义特征工程**：利用 `roberta-base-finetuned-jd-binary-chinese` 将 MD&A 长文本映射至语义空间，提取情感倾向、风险预警、不确定性等多维度特征。后续通过 PCA 降维，保留关键语义信息。
* **数据脱敏与中性化**：对预测目标进行 **行业剥离 (Industry De-exposured)**，即$Y = R_{stock} - R_{industry\_avg}$，以剔除行业贝塔干扰。

### 3. 稳健建模

* **LTR 框架**：使用 `XGBoost` 的 `rank:pairwise` 目标函数进行截面排序优化，提升 Top-K 标的选取的稳健性。
* **幸存者偏差控制**：样本池包含历史已退市公司，确保回测结果符合真实交易环境。

---

## 系统架构

```text
├── src/
│   ├── data_engine/      # 异步爬虫、公告日对齐、行情对齐
│   ├── smart_parser/     # 结构化 PDF 解析与 MD&A 提取
│   ├── features/         # 语义特征提取与 PCA 降维
│   ├── learning/         # XGBoost Ranker 训练、Optuna 贝叶斯搜索
│   ├── backtest/         # 考虑交易摩擦的 Alpha 收益回测
│   └── research/         # 特征重要性与 SHAP 可解释性分析
└── configs/              # 行业分类映射与模型超参

```

---

## 核心技术栈

| 模块 | 核心技术 | 解决痛点 |
| --- | --- | --- |
| **PDF Parsing** | `PyMuPDF` | 解决表格噪音与跨页文本断裂 |
| **NLP Engine** | `roberta-base-finetuned-jd-binary-chinese` | 捕捉金融语境下的微弱情感信号 |
| **Ensemble** | `XGBoost` (LambdaMART) | 捕捉财务数据与语义特征的非线性交互 |
| **Strategy** | `VectorBT` / `Backtrader` | 计算 Sharpe, Sortino, Information Ratio |

---

## 快速开始

1. **环境配置**：
```bash
pip install -r requirements.txt

```

2. **启动端到端 Pipeline**：
```bash
python main_pipeline.py --mode full --start_year 2021 --industry_neutral True

```

---

## 实验产出

* **特征重要性分布**：展示语义特征（如“不确定性”）对超额收益的解释力度。
* **Long-Short 净值曲线**：展示相对于中证 800 指数的超额收益及最大回撤控制。

---

**免责声明**：本项目仅供学术研究使用。

---