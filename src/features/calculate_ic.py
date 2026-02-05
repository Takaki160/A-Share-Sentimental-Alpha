import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path

# ================= 工业级配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "FEATURE_FILE": DATA_DIR / "features_pca_embeddings.csv",
    "LABEL_FILE": DATA_DIR / "master_alignment_table.csv",
    "IC_OUTPUT_CSV": OUTPUT_DIR / "ic_analysis_results.csv",
    "IC_PLOT_PNG": OUTPUT_DIR / "ic_analysis_plot.png",
    
    # 核心列名配置
    "KEY_SYMBOL": "symbol",
    "KEY_YEAR": "report_year",
    "KEY_TARGET": "future_return",  # 60天涨跌幅列
    
    "MIN_SAMPLE_SIZE": 30  # 最小统计样本量
}

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# ===============================================

def load_and_clean_data():
    """加载数据并进行严格的类型清洗"""
    if not CONFIG["FEATURE_FILE"].exists():
        logging.error(f"特征文件缺失: {CONFIG['FEATURE_FILE']}")
        sys.exit(1)
    
    if not CONFIG["LABEL_FILE"].exists():
        logging.error(f"标签文件缺失: {CONFIG['LABEL_FILE']}")
        sys.exit(1)

    # 1. 强制 Symbol 为字符串，保留前导零 (如 000001)
    df_feat = pd.read_csv(CONFIG["FEATURE_FILE"], dtype={'symbol': str})
    df_label = pd.read_csv(CONFIG["LABEL_FILE"], dtype={'symbol': str})
    
    # 2. 检查列名是否存在
    if CONFIG["KEY_TARGET"] not in df_label.columns:
        logging.error(f"在标签文件中未找到列: [{CONFIG['KEY_TARGET']}]")
        logging.error(f"现有列名: {list(df_label.columns)}")
        sys.exit(1)

    # 3. 统一列名 (防止不同文件年份列名不一致)
    # 特征表通常有 'report_year' 或 'year'，这里做一次标准化检查
    feat_cols = df_feat.columns
    if 'report_year' not in feat_cols and 'year' in feat_cols:
        df_feat.rename(columns={'year': 'report_year'}, inplace=True)
        
    return df_feat, df_label

def calculate_ic_metrics(df_merged, feature_cols):
    """核心计算逻辑：计算 RankIC (斯皮尔曼) 和 PearsonIC"""
    ic_data = []
    target_col = CONFIG["KEY_TARGET"]
    
    for feat in feature_cols:
        # 提取当前因子的非空数据
        sub_df = df_merged[[feat, target_col]].dropna()
        
        # 样本量不足则跳过
        if len(sub_df) < CONFIG["MIN_SAMPLE_SIZE"]:
            continue
            
        # 1. Rank IC (斯皮尔曼相关) - 量化核心指标
        # 衡量因子的排名预测能力，抗异常值干扰
        rank_ic, p_value = stats.spearmanr(sub_df[feat], sub_df[target_col])
        
        # 2. Pearson IC (线性相关)
        pearson_ic, _ = stats.pearsonr(sub_df[feat], sub_df[target_col])
        
        ic_data.append({
            'Feature': feat,
            'Rank_IC': rank_ic,       # 排序依据
            'Pearson_IC': pearson_ic,
            'IC_Abs': abs(rank_ic),   #用于看相关性强度(不论方向)
            'P_Value': p_value,
            'Significant': p_value < 0.05 # 95% 置信度
        })
        
    if not ic_data:
        return pd.DataFrame()
        
    return pd.DataFrame(ic_data).sort_values('Rank_IC', ascending=False)

def plot_ic_chart(df_ic):
    """绘制工业级 IC 条形图"""
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # A股习惯：红涨绿跌 (正相关为红，负相关为绿)
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in df_ic['Rank_IC']]
    
    ax = sns.barplot(x='Feature', y='Rank_IC', data=df_ic, palette=colors, hue='Rank_IC', legend=False)
    
    # 图表修饰
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'PCA Features Predictive Power (Target: {CONFIG["KEY_TARGET"]})', fontsize=16, fontweight='bold')
    plt.ylabel('Rank IC (Spearman Correlation)', fontsize=12)
    plt.xlabel('Latent Semantic Features (PCA)', fontsize=12)
    plt.xticks(rotation=45)
    
    # 标注显著性 (* 号)
    for i, p_val in enumerate(df_ic['P_Value']):
        if p_val < 0.05:
            height = df_ic.iloc[i]['Rank_IC']
            offset = 0.005 if height > 0 else -0.015
            ax.text(i, height + offset, '*', ha='center', color='black', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CONFIG["IC_PLOT_PNG"], dpi=300)
    plt.close()

def main():
    logging.info("启动因子有效性分析 (IC Test)...")
    
    # 1. 数据加载
    df_feat, df_label = load_and_clean_data()
    logging.info(f"特征数据: {df_feat.shape}, 标签数据: {df_label.shape}")
    
    # 2. 数据对齐 (Inner Merge)
    # 基于 股票代码 + 年份 严格对齐
    merge_keys = [CONFIG["KEY_SYMBOL"], CONFIG["KEY_YEAR"]]
    
    df_merged = pd.merge(
        df_feat, 
        df_label[merge_keys + [CONFIG["KEY_TARGET"]]], 
        on=merge_keys, 
        how='inner'
    )
    
    if df_merged.empty:
        logging.error("合并后数据为空！")
        logging.error("请检查两个文件的 'symbol' 和 'report_year' 格式是否完全一致。")
        return

    logging.info(f"对齐成功，有效样本数: {len(df_merged)}")
    
    # 3. 计算 IC
    pca_cols = [c for c in df_merged.columns if c.startswith('pca_')]
    logging.info(f"正在测试 {len(pca_cols)} 个 PCA 特征...")
    
    df_ic = calculate_ic_metrics(df_merged, pca_cols)
    
    if df_ic.empty:
        logging.warning("未能计算出有效的 IC (可能样本量太小或方差为0)")
        return

    # 4. 输出与保存
    print("\n" + "="*70)
    print(f"因子 IC 分析结果 (Target: {CONFIG['KEY_TARGET']})")
    print("* 标记为 P-Value < 0.05 (统计显著)")
    print("="*70)
    
    # 格式化输出，保留4位小数
    display_cols = ['Feature', 'Rank_IC', 'Pearson_IC', 'P_Value', 'Significant']
    print(df_ic[display_cols].round(4).to_string(index=False))
    
    df_ic.to_csv(CONFIG["IC_OUTPUT_CSV"], index=False)
    logging.info(f"结果已保存: {CONFIG['IC_OUTPUT_CSV']}")
    
    # 5. 绘图
    plot_ic_chart(df_ic)
    logging.info(f"图表已生成: {CONFIG['IC_PLOT_PNG']}")
    
    # 6. 简要结论
    best = df_ic.iloc[0]
    worst = df_ic.iloc[-1]
    print("快速解读:")
    print(f"最强正向因子: {best['Feature']} (IC={best['Rank_IC']:.4f})")
    print(f"最强负向因子: {worst['Feature']} (IC={worst['Rank_IC']:.4f})")

if __name__ == "__main__":
    main()