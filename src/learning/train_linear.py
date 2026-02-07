import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCAN_RESULT_FILE = REPORTS_DIR / "single_factor_scan_results.csv"

CONFIG = {
    "FEATURE_FILE": DATA_DIR / "features_pca_embeddings.csv",
    "LABEL_FILE": DATA_DIR / "master_alignment_table.csv",
    
    # === 关键设置 ===
    "TOP_N_FEATURES": 10,  # 只使用最强的 10 个因子
    
    "TEST_START_DATE": "2023-01-01",
    "TEST_END_DATE": "2024-12-31",
    "ROLLING_MONTHS": 1,
    
    "MIN_TRAIN_SAMPLES": 200,
    "MIN_TEST_SAMPLES": 20,
    "RIDGE_ALPHA": 0.01,
    "TARGET_COL": "future_return",
    "DATE_COL": "disclosure_date"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# ===============================================

def get_top_features():
    """读取扫描报告，获取 IC 绝对值最大的 Top N 特征"""
    if not SCAN_RESULT_FILE.exists():
        logging.error(f"❌ Scan result not found at {SCAN_RESULT_FILE}. Please run scan_best_factor.py first!")
        sys.exit(1)
        
    df_scan = pd.read_csv(SCAN_RESULT_FILE)
    
    # 鲁棒性检查：如果 CSV 里没有 Abs_IC 列，现场计算
    if 'Abs_IC' not in df_scan.columns:
        if 'Rank_IC_Mean' in df_scan.columns:
            df_scan['Abs_IC'] = df_scan['Rank_IC_Mean'].abs()
        else:
            logging.error("❌ Invalid CSV format: 'Rank_IC_Mean' column missing.")
            sys.exit(1)
        
    # 按绝对值排序，取 Top N
    top_df = df_scan.sort_values('Abs_IC', ascending=False).head(CONFIG["TOP_N_FEATURES"])
    top_features = top_df['Feature'].tolist()
    
    logging.info(f"💎 Selected Top {len(top_features)} Features based on |IC|:")
    logging.info(f"   {top_features}")
    return top_features

def load_data(selected_features):
    logging.info("Loading and aligning data...")
    if not CONFIG["FEATURE_FILE"].exists():
        logging.error(f"Feature file missing: {CONFIG['FEATURE_FILE']}")
        sys.exit(1)

    df_feat = pd.read_csv(CONFIG["FEATURE_FILE"], dtype={'symbol': str})
    df_label = pd.read_csv(CONFIG["LABEL_FILE"], dtype={'symbol': str})
    df_label[CONFIG["DATE_COL"]] = pd.to_datetime(df_label[CONFIG["DATE_COL"]])
    
    df = pd.merge(df_label, df_feat, on=['symbol', 'report_year'], how='inner')
    
    # 过滤 NaN
    original_len = len(df)
    df = df.dropna(subset=[CONFIG["TARGET_COL"]] + selected_features)
    if len(df) < original_len:
        logging.info(f"Dropped {original_len - len(df)} rows with NaNs.")
        
    df = df.sort_values(CONFIG["DATE_COL"]).reset_index(drop=True)
    return df

def run_backtest(df, features):
    results = []
    # 用于存储模型系数，验证是否正确反转了负因子
    last_coefficients = None 
    
    current_date = pd.to_datetime(CONFIG["TEST_START_DATE"])
    end_date = pd.to_datetime(CONFIG["TEST_END_DATE"])
    
    logging.info(f"Starting Rolling Backtest ({current_date.date()} -> {end_date.date()})")
    
    while current_date <= end_date:
        next_date = current_date + relativedelta(months=CONFIG["ROLLING_MONTHS"])
        
        # PIT 切分
        mask_train = df[CONFIG["DATE_COL"]] < current_date
        mask_test = (df[CONFIG["DATE_COL"]] >= current_date) & (df[CONFIG["DATE_COL"]] < next_date)
        
        df_train = df[mask_train]
        df_test = df[mask_test]
        
        if len(df_train) < CONFIG["MIN_TRAIN_SAMPLES"]:
            current_date = next_date
            continue
        if len(df_test) < CONFIG["MIN_TEST_SAMPLES"]:
            logging.warning(f"[{current_date.strftime('%Y-%m')}] Skip: Insufficient test samples ({len(df_test)})")
            current_date = next_date
            continue
            
        X_train = df_train[features].values
        y_train = df_train[CONFIG["TARGET_COL"]].values
        X_test = df_test[features].values
        y_test = df_test[CONFIG["TARGET_COL"]].values
        
        # === 核心处理 ===
        # 1. Target Ranking (将收益率转为 0~1)
        y_train_rank = rankdata(y_train) / len(y_train)
        
        # 2. Standardization (Z-Score)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Ridge Regression
        model = Ridge(alpha=CONFIG["RIDGE_ALPHA"], random_state=42)
        model.fit(X_train_scaled, y_train_rank)
        
        # 保存最后一期的系数以便检查
        last_coefficients = model.coef_
        
        # 4. Predict
        preds = model.predict(X_test_scaled)
        
        # 5. Metrics
        rank_ic, _ = spearmanr(preds, y_test)
        
        # 简单计算多空收益 (Top 20% - Bottom 20%)
        test_df_slice = df_test.copy()
        test_df_slice['pred'] = preds
        # 分5组
        try:
            test_df_slice['group'] = pd.qcut(test_df_slice['pred'], 5, labels=False, duplicates='drop')
            long_ret = test_df_slice[test_df_slice['group'] == 4][CONFIG["TARGET_COL"]].mean()
            short_ret = test_df_slice[test_df_slice['group'] == 0][CONFIG["TARGET_COL"]].mean()
            ls_ret = long_ret - short_ret
        except ValueError:
            ls_ret = 0.0
        
        results.append({
            "Month": current_date,
            "Rank_IC": rank_ic,
            "Long_Short_Ret": ls_ret,
            "Sample_Size": len(df_test)
        })
        
        logging.info(f"[{current_date.strftime('%Y-%m')}] IC: {rank_ic:.4f} | L-S Ret: {ls_ret:.2%} (n={len(df_test)})")
        current_date = next_date
        
    return pd.DataFrame(results), last_coefficients

def main():
    print(f"🚀 Starting Linear Model (Top {CONFIG['TOP_N_FEATURES']} Factors)")
    
    # 1. 选因子
    top_features = get_top_features()
    
    # 2. 读数据
    df = load_data(top_features)
    
    # 3. 跑回测
    df_res, last_coefs = run_backtest(df, top_features)
    
    if df_res.empty:
        logging.error("No results generated.")
        return
        
    # 4. 统计与打印
    mean_ic = df_res['Rank_IC'].mean()
    ic_ir = mean_ic / df_res['Rank_IC'].std() if df_res['Rank_IC'].std() != 0 else 0
    cum_ls_ret = df_res['Long_Short_Ret'].sum()
    
    print("\n" + "="*60)
    print(f"📊 LINEAR MODEL STRATEGY RESULTS")
    print("="*60)
    print(f"Factors Used : {len(top_features)}")
    print(f"Mean Rank IC : {mean_ic:.4f}")
    print(f"IC IR        : {ic_ir:.4f}")
    print(f"Cum L-S Ret  : {cum_ls_ret:.2%}")
    print("-" * 60)
    
    # --- 关键检查：打印系数 ---
    print("\n🧐 Model Coefficients (Check Factor Direction):")
    if last_coefs is not None:
        coef_df = pd.DataFrame({'Feature': top_features, 'Coef': last_coefs})
        coef_df = coef_df.sort_values('Coef') # 按系数从小到大排
        print(coef_df.to_string(index=False))
        print("\n(Note: Negative Coef means the model correctly reversed the negative factor.)")
    print("="*60)

    # 5. 绘图
    df_res['Cumulative_IC'] = df_res['Rank_IC'].cumsum()
    df_res['Cumulative_LS_Ret'] = df_res['Long_Short_Ret'].cumsum()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制 IC (左轴)
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative IC', color=color)
    ax1.plot(df_res['Month'], df_res['Cumulative_IC'], color=color, linewidth=2, label='Cum IC')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 绘制多空收益 (右轴)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative Long-Short Return', color=color)
    ax2.plot(df_res['Month'], df_res['Cumulative_LS_Ret'], color=color, linewidth=2, linestyle='--', label='Cum L-S Ret')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Linear Model: Top {CONFIG["TOP_N_FEATURES"]} PCA Features')
    fig.tight_layout()
    
    plot_path = REPORTS_DIR / "backtest_linear.png"
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()