# A-Share-Sentimental-Alpha: 基于年报语义分析的 A 股收益率预测系统

## 📌 项目概述

本项目是一套完整的**非结构化数据量化投资方案**。系统通过自动化爬取中国 A 股上市公司年度报告（PDF），利用深度学习模型（FinBERT）提取“管理层讨论与分析（MD&A）”章节的语义特征，并结合传统财务因子，构建梯度提升决策树（XGBoost）模型预测股票次年超额收益。

**核心目标**：验证年报文本中的非对称信息（情感倾向、语调变化、语言复杂性）是否能作为 Alpha 信号提供超越大盘的预测能力。

---

## 🛠️ 技术架构

系统分为五大核心流水线：

1. **数据采集 (Data Pipeline)**:
* 基于 `AkShare` 获取 A 股历史成分股及财务摘要。
* 异步并发爬虫从巨潮资讯（CNINFO）抓取数万份年度报告 PDF。


2. **智能解析 (Smart Parser)**:
* 结合正则表达式与 layout 分析，精准定位并提取 **MD&A（管理层讨论与分析）** 章节。
* 处理 PDF 跨页断句、剔除表格及页眉页脚噪声。


3. **特征工程 (Feature Engineering)**:
* **语义特征**: 利用 `FinBERT-Chinese` 提取 768 维文本向量及情感概率分布（积极/中性/消极）。
* **差异特征**: 计算相邻年份 Embedding 的余弦相似度（Cosine Similarity），捕捉经营策略的漂移。
* **量化因子**: 集成多因子模型（Size, Value, Momentum, Quality）作为控制变量。


4. **建模与训练 (Machine Learning)**:
* **模型**: XGBoost / LightGBM。
* **验证**: 采用 **Walk-forward Validation（滚动时序验证）**，严防“回看偏误（Look-ahead bias）”。


5. **回测引擎 (Strategy Backtest)**:
* 构建 **Long-Short 组合策略**。
* 计算核心指标：Annualized Return, Sharpe Ratio, Max Drawdown, Information Ratio。



---

## 📂 目录结构

```text
├── src/
│   ├── crawler/          # 异步爬虫模块
│   ├── parser/           # PDF解析与MD&A定位逻辑
│   ├── features/         # 文本向量化与因子对齐
│   ├── models/           # 训练脚本与超参数优化(Optuna)
│   └── backtest/         # 净值计算与可视化
├── configs/              # 策略参数配置文件
├── notebooks/            # 特征探索性分析(EDA)
└── requirements.txt      # 环境依赖

```

---

## 🚀 核心亮点 (Key Features)

* **金融级语义理解**：弃用简单的词频统计（TF-IDF），采用针对中文金融语境微调的 **FinBERT** 模型，能识别“业绩承压”、“稳中向好”等短语背后的深层情感。
* **鲁棒的解析方案**：针对 A 股年报排版多变的痛点，开发了基于锚点定位的章节提取算法， MD&A 提取准确率达 90%+。
* **端到端 Pipeline**：实现从原始 PDF 下载到回测报告生成的全自动化链路。
* **可解释性分析**：引入 **SHAP (SHapley Additive exPlanations)** 值，分析哪些文本关键词对股价预测起到了决定性作用。

---

## 📊 预期产出

* **因子有效性分析**：文本特征与收益率的相关性热力图。
* **回测净值曲线**：模型策略 vs 沪深300 指数的收益对比图。
* **特征贡献排行**：展示模型中最具影响力的财务因子与文本因子。

---

## 📝 免责声明

本项目仅用于学术研究与技术演示，不构成任何投资建议。股市有风险，入市需谨慎。
