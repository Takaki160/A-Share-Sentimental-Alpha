import sys
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler

# 修复 Windows 控制台
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCAN_RESULT_FILE = REPORTS_DIR / "single_factor_scan_results.csv"  # 读取扫描结果

CONFIG = {
    "FEATURE_FILE": DATA_DIR / "features_pca_embeddings.csv",
    "LABEL_FILE": DATA_DIR / "master_alignment_table.csv",
    
    # === 关键：只用 Top 10 ===
    "TOP_N_FEATURES": 10,
    
    # 回测设置
    "TEST_START_DATE": "2023-01-01",
    "TEST_END_DATE": "2024-12-31",
    "ROLLING_MONTHS": 1,
    
    # 数据控制
    "MIN_TRAIN_SAMPLES": 200,
    "MIN_TEST_SAMPLES": 20,
    "VALIDATION_RATIO": 0.15,
    
    "TARGET_COL": "future_return",
    "DATE_COL": "disclosure_date",
    
    # LightGBM 参数 (保持你提供的配置)
    "LGBM_PARAMS": {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.03,
        'num_leaves': 8,       # 针对少特征(10个)进行了限制，防止过拟合
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ===============================================

def get_top_features():
    """读取扫描报告，获取 Top N 特征"""
    if not SCAN_RESULT_FILE.exists():
        logging.error(f"Scan result not found at {SCAN_RESULT_FILE}. Run scan_best_factor.py first!")
        sys.exit(1)
        
    df_scan = pd.read_csv(SCAN_RESULT_FILE)
    
    # 鲁棒性处理
    if 'Abs_IC' not in df_scan.columns:
        df_scan['Abs_IC'] = df_scan['Rank_IC_Mean'].abs()
        
    # 按绝对值排序
    top_features = df_scan.sort_values('Abs_IC', ascending=False).head(CONFIG["TOP_N_FEATURES"])['Feature'].tolist()
    
    logging.info(f"💎 Selected Top {CONFIG['TOP_N_FEATURES']} Features for LightGBM:")
    logging.info(f"   {top_features}")
    return top_features

def load_data(selected_features):
    logging.info("Loading data...")
    df_feat = pd.read_csv(CONFIG["FEATURE_FILE"], dtype={'symbol': str})
    df_label = pd.read_csv(CONFIG["LABEL_FILE"], dtype={'symbol': str})
    df_label[CONFIG["DATE_COL"]] = pd.to_datetime(df_label[CONFIG["DATE_COL"]])
    
    df = pd.merge(df_label, df_feat, on=['symbol', 'report_year'], how='inner')
    
    # 只保留选中的特征
    df = df.dropna(subset=[CONFIG["TARGET_COL"]] + selected_features)
    df = df.sort_values(CONFIG["DATE_COL"]).reset_index(drop=True)
    
    return df

def run_backtest(df, features):
    results = []
    feature_importance_acc = np.zeros(len(features))
    model_count = 0
    
    current_date = pd.to_datetime(CONFIG["TEST_START_DATE"])
    end_date = pd.to_datetime(CONFIG["TEST_END_DATE"])
    
    logging.info(f"Starting Optimized LGBM Rolling ({current_date.date()} -> {end_date.date()})")
    
    while current_date <= end_date:
        next_date = current_date + relativedelta(months=CONFIG["ROLLING_MONTHS"])
        
        mask_train = df[CONFIG["DATE_COL"]] < current_date
        mask_test = (df[CONFIG["DATE_COL"]] >= current_date) & (df[CONFIG["DATE_COL"]] < next_date)
        
        df_train_full = df[mask_train].copy()
        df_test = df[mask_test].copy()
        
        if len(df_train_full) < CONFIG["MIN_TRAIN_SAMPLES"]:
            current_date = next_date
            continue
        if len(df_test) < CONFIG["MIN_TEST_SAMPLES"]:
            logging.warning(f"[{current_date.strftime('%Y-%m')}] Skip: Insufficient samples ({len(df_test)})")
            current_date = next_date
            continue
            
        # 验证集切分
        split_idx = int(len(df_train_full) * (1 - CONFIG["VALIDATION_RATIO"]))
        
        X_train = df_train_full.iloc[:split_idx][features]
        y_train = df_train_full.iloc[:split_idx][CONFIG["TARGET_COL"]]
        
        X_valid = df_train_full.iloc[split_idx:][features]
        y_valid = df_train_full.iloc[split_idx:][CONFIG["TARGET_COL"]]
        
        # === 核心处理: Target Ranking ===
        # 将 Y 转换为 0-1 排名，与 Linear 逻辑保持一致，剔除 Beta
        y_train = rankdata(y_train) / len(y_train)
        y_valid = rankdata(y_valid) / len(y_valid)
        
        X_test = df_test[features]
        y_test = df_test[CONFIG["TARGET_COL"]]
        
        # 训练 LightGBM
        model = lgb.LGBMRegressor(**CONFIG["LGBM_PARAMS"])
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse',
            callbacks=callbacks
        )
        
        # 预测
        preds = model.predict(X_test)
        
        # 计算 IC
        rank_ic, _ = spearmanr(preds, y_test)
        
        # 计算多空收益 (Long - Short)
        test_df_slice = df_test.copy()
        test_df_slice['pred'] = preds
        try:
            test_df_slice['group'] = pd.qcut(test_df_slice['pred'], 5, labels=False, duplicates='drop')
            long_ret = test_df_slice[test_df_slice['group'] == 4][CONFIG["TARGET_COL"]].mean()
            short_ret = test_df_slice[test_df_slice['group'] == 0][CONFIG["TARGET_COL"]].mean()
            ls_ret = long_ret - short_ret
        except ValueError:
            ls_ret = 0.0
        
        # 记录重要性
        feature_importance_acc += model.feature_importances_
        model_count += 1
        
        results.append({
            "Month": current_date,
            "Rank_IC": rank_ic,
            "Long_Short_Ret": ls_ret,
            "Sample_Size": len(df_test)
        })
        
        logging.info(f"[{current_date.strftime('%Y-%m')}] IC: {rank_ic:.4f} | L-S Ret: {ls_ret:.2%} (n={len(df_test)})")
        current_date = next_date
        
    return pd.DataFrame(results), feature_importance_acc, model_count

def main():
    print(f"🚀 Starting LightGBM (Top {CONFIG['TOP_N_FEATURES']} Factors)")
    
    # 1. 自动选因子
    top_features = get_top_features()
    
    # 2. 载入数据
    df = load_data(top_features)
    
    # 3. 回测
    df_res, feat_imp_sum, model_cnt = run_backtest(df, top_features)
    
    if df_res.empty:
        logging.error("No results generated.")
        return

    # 4. 统计
    mean_ic = df_res['Rank_IC'].mean()
    ic_ir = mean_ic / df_res['Rank_IC'].std() if df_res['Rank_IC'].std() != 0 else 0
    cum_ls_ret = df_res['Long_Short_Ret'].sum()
    
    print("\n" + "="*60)
    print(f"📊 LIGHTGBM RESULTS")
    print("="*60)
    print(f"Factors Used : {len(top_features)}")
    print(f"Mean Rank IC : {mean_ic:.4f}")
    print(f"IC IR        : {ic_ir:.4f}")
    print(f"Cum L-S Ret  : {cum_ls_ret:.2%}")
    print("="*60)
    
    # 5. 绘图 (双轴: IC + L-S Ret)
    df_res['Cumulative_IC'] = df_res['Rank_IC'].cumsum()
    df_res['Cumulative_LS_Ret'] = df_res['Long_Short_Ret'].cumsum()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative IC', color=color)
    ax1.plot(df_res['Month'], df_res['Cumulative_IC'], color=color, linewidth=2, label='Cum IC')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Cumulative Long-Short Return', color=color)
    ax2.plot(df_res['Month'], df_res['Cumulative_LS_Ret'], color=color, linewidth=2, linestyle='--', label='Cum L-S Ret')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'LightGBM: Top {CONFIG["TOP_N_FEATURES"]} Features')
    fig.tight_layout()
    
    plot_path = REPORTS_DIR / "backtest_lgbm.png"
    plt.savefig(plot_path, dpi=300)
    
    # 6. 绘制特征重要性
    if model_cnt > 0:
        avg_imp = feat_imp_sum / model_cnt
        df_imp = pd.DataFrame({'Feature': top_features, 'Importance': avg_imp}).sort_values('Importance', ascending=False)
        print("\n🧐 Feature Importance (LightGBM):")
        print(df_imp.to_string(index=False))

    logging.info(f"Plots saved to reports/backtest_lgbm.png")

if __name__ == "__main__":
    main()