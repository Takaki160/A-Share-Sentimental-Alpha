# A-Share-Sentimental-Alpha

A-Share-Sentimental-Alpha 是一个端到端的量化研究框架，旨在挖掘中国 A 股上市公司年度报告中“管理层讨论与分析”（MD&A）章节的文本情绪价值。通过结合预训练语言模型与机器学习模型，本项目尝试从非结构化文本中提取 Alpha 因子，并预测个股年报披露后 T+60 日收益。

## 🚀 核心特性

- **数据源**: 整合 AkShare 年报接口与 Tushare 股价接口，构建完整的“文本-行情”对齐库。
- **文本提取**: 基于 PyMuPDF 与正则表达式，解析并提取年报中 MD&A 章节内容。
- **向量化**: 采用 `Chinese-RoBERTa-WWM-Ext` 模型生成文本 Embedding。
- **特征工程**: 利用 PCA 优化高维特征，保留核心语义方差。
- **模型预测**: 根据线性 (Elastic Net) 与非线性 (LightGBM) 两种基准模型，预测 T+60 日收益。

## 📂 项目结构

```text
├── data/                    # 原始 PDF、清洗后的 CSV 及特征矩阵
├── results/                 # 模型预测结果与评价指标 (JSON/CSV)
├── src/
│   ├── data_engine/         # 数据采集、下载及披露日期对齐
│   ├── text_parser/         # PDF 解析与 MD&A 章节文本提取
│   ├── feature_builder/     # RoBERTa 语义向量化及 PCA 降维
│   └── models/              # 线性/非线性回归模型训练与回测
├── .log                     # 日志文件
└── README.md
```

## 🛠️ 技术流程

1.  **数据采集 (Data Ingestion)**:
    - `aligner`获取上市公司 PDF 年报链接，对齐定期报告披露预约日、实际日及后 60 个交易日的收盘价数据。
    - `pdf_downloader`下载 PDF 年报文件。
    - `clean_data`剔除无效 PDF，如大小过小或内容为空。
2.  **文本挖掘 (Text Mining)**:
    - `pdf_parser`: 将 PDF 转为结构化纯文本。
    - `mda_extractor`: 定位“管理层讨论与分析”章节，过滤无效文本。
3.  **特征化 (Feature Engineering)**:
    - `mda_embedding_pca` 加载 [Chinese BERT with Whole Word Masking](https://huggingface.co/hfl/chinese-roberta-wwm-ext) 模型进行语义编码，并对生成的高维 Embedding 进行 PCA 降维。
4.  **建模与评估 (Modeling & Backtest)**:
    - **数据集划分**: 2021-2022 (训练集), 2023 (验证集), 2024 (测试集)。
    - **模型应用**: 线性回归`elastic_net`与非线性回归`lightgbm_model`。
    - **评估指标**: 采用 Rank IC, RMSE, MAE 评估模型性能。

## 🏁 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 配置 API Token
在 `src/data_engine/aligner.py` 中填入你的 Tushare Token。

### 3. 运行全流程
```bash
# 1. 下载数据
python src/data_engine/aligner.py
python src/data_engine/pdf_downloader.py
python src/data_engine/clean_data.py
# 2. 提取文本
python src/text_parser/pdf_parser.py
python src/text_parser/mda_extractor.py
# 3. 特征提取 (建议使用 GPU)
python src/feature_builder/mda_embedding_pca.py
# 4. 模型训练
python src/models/elastic_net.py
python src/models/lightgbm_model.py
```

## 📈 预期目标
通过 MD&A 文本挖掘，捕捉管理层在正式公告中流露的经营态度（如积极、稳健或回避风险），从而构建具备统计学显著意义的文本 Alpha 因子。

## ⚠️ 免责声明
本项目仅供量化研究与交流使用，不构成任何投资建议。投资有风险，入市需谨慎。
