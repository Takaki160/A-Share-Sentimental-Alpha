import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, ttest_1samp

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "FEATURE_FILE": DATA_DIR / "features_pca_embeddings.csv",
    "LABEL_FILE": DATA_DIR / "master_alignment_table.csv",
    
    # 测试区间
    "TEST_START_DATE": "2021-01-01",
    "TEST_END_DATE": "2022-12-31",
    
    # 数据控制
    "MIN_TEST_SAMPLES": 20,
    "TARGET_COL": "future_return",
    "DATE_COL": "disclosure_date",
    
    # 分组参数
    "QUANTILES": 10  # 分成10组 (Top 10% vs Bottom 10%)
}

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ===============================================

def load_and_clean_data():
    """加载并清洗数据，直接过滤出回测区间"""
    logging.info("Loading data...")
    if not CONFIG["FEATURE_FILE"].exists() or not CONFIG["LABEL_FILE"].exists():
        logging.error("Missing input files.")
        return None, None

    df_feat = pd.read_csv(CONFIG["FEATURE_FILE"], dtype={'symbol': str})
    df_label = pd.read_csv(CONFIG["LABEL_FILE"], dtype={'symbol': str})
    df_label[CONFIG["DATE_COL"]] = pd.to_datetime(df_label[CONFIG["DATE_COL"]])
    
    # 合并
    df = pd.merge(df_label, df_feat, on=['symbol', 'report_year'], how='inner')
    
    # 筛选时间区间
    start_dt = pd.to_datetime(CONFIG["TEST_START_DATE"])
    end_dt = pd.to_datetime(CONFIG["TEST_END_DATE"])
    df = df[(df[CONFIG["DATE_COL"]] >= start_dt) & (df[CONFIG["DATE_COL"]] <= end_dt)]
    
    # 提取 PCA 列名
    pca_cols = [c for c in df.columns if c.startswith('pca_')]
    
    # 去除 NaN
    df = df.dropna(subset=[CONFIG["TARGET_COL"]] + pca_cols)
    
    logging.info(f"Data Loaded: {len(df)} samples, {len(pca_cols)} features.")
    return df, pca_cols

def analyze_single_factor(df_grouped, feature_name):
    """分析单个因子的表现"""
    ic_list = []
    long_short_ret_list = []
    
    for date, group in df_grouped:
        if len(group) < CONFIG["MIN_TEST_SAMPLES"]:
            continue
            
        # 1. 计算 Rank IC
        # 注意：这里我们保留原始方向，如果是负IC，说明因子是反向指标
        ic, _ = spearmanr(group[feature_name], group[CONFIG["TARGET_COL"]])
        ic_list.append(ic)
        
        # 2. 计算分组收益 (Top - Bottom)
        # pd.qcut 将数据分为 N 组，label=False 使组号为 0,1,2,3,4
        try:
            group['quantile'] = pd.qcut(group[feature_name], CONFIG["QUANTILES"], labels=False, duplicates='drop')
            
            # 多头：最大的一组 (4)
            # 空头：最小的一组 (0)
            long_ret = group[group['quantile'] == (CONFIG["QUANTILES"] - 1)][CONFIG["TARGET_COL"]].mean()
            short_ret = group[group['quantile'] == 0][CONFIG["TARGET_COL"]].mean()
            
            long_short_ret_list.append(long_ret - short_ret)
        except ValueError:
            # 样本太少无法分位时跳过
            continue

    if not ic_list:
        return None

    # --- 统计指标 ---
    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list)
    ic_ir = ic_mean / ic_std if ic_std != 0 else 0
    
    # IC T-Value (检验 IC 是否显著不为0)
    # t = mean / (std / sqrt(n))
    t_stat, _ = ttest_1samp(ic_list, 0)
    
    ls_mean = np.mean(long_short_ret_list)
    ls_win_rate = np.mean([r > 0 for r in long_short_ret_list])

    return {
        "Feature": feature_name,
        "Rank_IC_Mean": ic_mean,
        "Rank_IC_IR": ic_ir,
        "IC_T_Stat": t_stat,          # 绝对值 > 2 说明显著
        "Long_Short_Ret": ls_mean,    # 多空组合平均收益
        "Win_Rate": ls_win_rate       # 胜率
    }

def main():
    print(f"🔎 Scanning Best Factors ({CONFIG['TEST_START_DATE']} - {CONFIG['TEST_END_DATE']})...")
    
    df, pca_cols = load_and_clean_data()
    if df is None or df.empty:
        logging.error("No data available.")
        return

    # 优化：先按日期分组，避免在循环中重复分组
    # list(grouped) 将生成器转为列表，方便多次复用
    df_grouped = list(df.groupby(CONFIG["DATE_COL"]))
    
    results = []
    
    # 遍历所有特征
    for feat in pca_cols:
        res = analyze_single_factor(df_grouped, feat)
        if res:
            results.append(res)
            
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        logging.warning("No valid results computed.")
        return

    # 增加一列 Abs_IC 便于筛选强度
    df_res['Abs_IC'] = df_res['Rank_IC_Mean'].abs()
    
    # 排序：按 IC 绝对值排序
    df_sorted = df_res.sort_values("Abs_IC", ascending=False).reset_index(drop=True)
    
    # 保存结果
    output_path = REPORTS_DIR / "single_factor_scan_results.csv"
    df_sorted.to_csv(output_path, index=False)
    
    # --- 打印报告 ---
    print("\n" + "="*80)
    print(f"🏆 Top 10 Most Significant Factors (Ranked by |IC|)")
    print("="*80)
    print(f"{'Feature':<10} | {'IC Mean':<10} | {'IC IR':<8} | {'T-Stat':<8} | {'L-S Ret':<10} | {'Win Rate':<8}")
    print("-" * 80)
    
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['Feature']:<10} | {row['Rank_IC_Mean']:>8.4f}   | {row['Rank_IC_IR']:>6.2f}   | {row['IC_T_Stat']:>6.2f}   | {row['Long_Short_Ret']:>8.4%}   | {row['Win_Rate']:>7.2%}")
        
    print("="*80)
        
    logging.info(f"Full report saved to: {output_path}")

if __name__ == "__main__":
    main()