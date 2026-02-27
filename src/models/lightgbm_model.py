import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, mstats
import json

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    "MASTER_CSV": DATA_DIR / "clean_master.csv",
    "FEATURES_CSV": DATA_DIR / "features_pca_embeddings.csv",
    "TARGET_COL": "future_return",
    
    "TRAIN_YEARS": [2021, 2022],
    "VAL_YEAR": 2023,
    "TEST_YEAR": 2024,
    
    "WINS_LIMITS": [0.01, 0.01],
    
    # LightGBM Parameters (Initial reasonable defaults for small features)
    "LGB_PARAMS": {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_child_samples": 20
    },
    
    "PREDS_OUTPUT": RESULTS_DIR / "lightgbm_preds_2024.csv",
    "SUMMARY_OUTPUT": RESULTS_DIR / "lightgbm_summary.json"
}

def cross_sectional_zscore(df, features_cols):
    """
    Apply cross-sectional z-score normalization per 'report_year' 
    to ensure proper centering and scaling without data leakage.
    """
    df0 = df.copy()
    df0[features_cols] = df.groupby('report_year')[features_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10) # Add small epsilon to avoid division by zero
    )
    return df0

def calculate_rank_ic(y_true, y_pred, groups):
    """
    Calculate the Cross-sectional Spearman correlation (Rank IC Mean).
    [FIXED] Now calculates IC per cross-section (group) to avoid time-series bias.
    """
    df_tmp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': groups})
    ic_list = []
    for _, group in df_tmp.groupby('group'):
        if np.unique(group['y_pred']).size > 1:
            ic_list.append(spearmanr(group['y_true'], group['y_pred'])[0])
    return np.mean(ic_list) if ic_list else -1.0

def main():
    print("--- LightGBM Pipeline ---")
    
    # 1. Load Data
    if not CONFIG["MASTER_CSV"].exists() or not CONFIG["FEATURES_CSV"].exists():
        print("Error: Required CSV files missing.")
        return

    df_master = pd.read_csv(CONFIG["MASTER_CSV"])
    df_features = pd.read_csv(CONFIG["FEATURES_CSV"])
    
    # Clean symbols (ensure 6-digit strings)
    df_master['symbol'] = df_master['symbol'].astype(str).str.zfill(6)
    df_features['symbol'] = df_features['symbol'].astype(str).str.zfill(6)
    
    # 2. Merge Data
    df = pd.merge(
        df_master[['symbol', 'report_year', CONFIG["TARGET_COL"]]], 
        df_features, 
        on=['symbol', 'report_year'], 
        how='inner'
    )
    
    # 3. Split Data into Train / Val / Test
    train_df = df[df['report_year'].isin(CONFIG["TRAIN_YEARS"])].copy()
    val_df   = df[df['report_year'] == CONFIG["VAL_YEAR"]].copy()
    test_df  = df[df['report_year'] == CONFIG["TEST_YEAR"]].copy()
    
    if len(val_df) == 0 or len(test_df) == 0:
        print("Error: Missing validation or test year data.")
        return
    
    # Apply Winsorization Cross-Sectionally on Train and Val sets separately to avoid data leakage
    def winsorize_target(group):
        return mstats.winsorize(group, limits=CONFIG["WINS_LIMITS"])

    train_df[CONFIG["TARGET_COL"]] = train_df.groupby('report_year')[CONFIG["TARGET_COL"]].transform(winsorize_target)
    val_df[CONFIG["TARGET_COL"]] = val_df.groupby('report_year')[CONFIG["TARGET_COL"]].transform(winsorize_target)
    
    features_cols = [c for c in df.columns if c.startswith('pca_')]

    train_df = cross_sectional_zscore(train_df, features_cols)
    val_df = cross_sectional_zscore(val_df, features_cols)
    test_df = cross_sectional_zscore(test_df, features_cols)
    
    X_train, y_train = train_df[features_cols], train_df[CONFIG["TARGET_COL"]]
    X_val, y_val     = val_df[features_cols], val_df[CONFIG["TARGET_COL"]]
    X_test, y_test   = test_df[features_cols], test_df[CONFIG["TARGET_COL"]]
    
    print(f"Data Split: Train({CONFIG['TRAIN_YEARS']}), Val({CONFIG['VAL_YEAR']}), Test({CONFIG['TEST_YEAR']})")

    # 4. Train LightGBM with Early Stopping on Validation Set
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print("Searching for best iteration on Validation Set (Early Stopping)...")
    model = lgb.train(
        CONFIG["LGB_PARAMS"],
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # 5. Out-of-Sample Evaluation (2024)
    print("Evaluating on 2024 Test Set with best iteration...")
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    
    test_rank_ic = calculate_rank_ic(y_test, test_preds, test_df['report_year']) # Pass grouping variable for cross-sectional IC calculation
    test_mse = mean_squared_error(y_test, test_preds)
    
    # 6. Save Results
    test_df['predicted_return'] = test_preds
    test_df[['symbol', 'report_year', CONFIG["TARGET_COL"], 'predicted_return']].to_csv(CONFIG["PREDS_OUTPUT"], index=False)
    
    summary = {
        "model": "LightGBM",
        "best_iteration": model.best_iteration,
        "test_rank_ic_mean": test_rank_ic,
        "test_mse": test_mse,
        "val_score": model.best_score['valid']['rmse'],
        "features_used": len(features_cols)
    }
    with open(CONFIG["SUMMARY_OUTPUT"], 'w') as f:
        json.dump(summary, f, indent=4)

    # 7. Final Output
    print("="*50)
    print(f"LIGHTGBM TEST PERFORMANCE (YEAR: {CONFIG['TEST_YEAR']})")
    print("="*50)
    print(f"Rank IC Mean (Spearman): {test_rank_ic:.6f}")
    print(f"MSE                    : {test_mse:.6f}")
    print(f"Best Iteration         : {model.best_iteration}")
    print(f"Summary saved to {CONFIG['SUMMARY_OUTPUT']}")
    print("="*50)

if __name__ == "__main__":
    main()