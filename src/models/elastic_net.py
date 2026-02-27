import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, mstats
import json
import warnings

# Suppress ConstantInputWarning from spearmanr when predictions are constant
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    "MASTER_CSV": DATA_DIR / "clean_master.csv",
    "FEATURES_CSV": DATA_DIR / "features_pca_embeddings.csv",
    "TARGET_COL": "future_return",
    
    # Sequential Time Split to avoid look-ahead bias
    "TRAIN_YEARS": [2021, 2022],  # Initial training
    "VAL_YEAR": 2023,             # Validation: Use this to tune hyperparameters
    "TEST_YEAR": 2024,            # Testing: Final out-of-sample evaluation
    
    "WINS_LIMITS": [0.01, 0.01],
    
    # Finer and smaller alpha range to avoid total feature suppression
    "PARAM_GRID": {
        "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01], 
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    
    "PREDS_OUTPUT": RESULTS_DIR / "elastic_net_preds_2024.csv",
    "SUMMARY_OUTPUT": RESULTS_DIR / "elastic_net_summary.json"
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
    print("--- Elastic Net Pipeline ---")
    
    # 1. Load Data
    if not CONFIG["MASTER_CSV"].exists() or not CONFIG["FEATURES_CSV"].exists():
        print("Error: Required CSV files missing.")
        return

    df_master = pd.read_csv(CONFIG["MASTER_CSV"])
    df_features = pd.read_csv(CONFIG["FEATURES_CSV"])
    
    df_master['symbol'] = df_master['symbol'].astype(str).str.zfill(6)
    df_features['symbol'] = df_features['symbol'].astype(str).str.zfill(6)
    
    df = pd.merge(
        df_master[['symbol', 'report_year', CONFIG["TARGET_COL"]]], 
        df_features, 
        on=['symbol', 'report_year'], 
        how='inner'
    )
    
    # 2. Split Data into Train / Val / Test
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

    # 4. Grid Search on Validation Set (2023)
    print("Searching for best hyperparameters on Validation Set (Rank IC)...")
    best_ic = -1.0
    best_params = {"alpha": 0.01, "l1_ratio": 0.5}

    for alpha in CONFIG["PARAM_GRID"]["alpha"]:
        for l1 in CONFIG["PARAM_GRID"]["l1_ratio"]:
            model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42, max_iter=10000)
            model.fit(X_train, y_train)
            
            val_preds = model.predict(X_val)
            
            ic = calculate_rank_ic(y_val, val_preds, val_df['report_year']) # Pass grouping variable for cross-sectional IC
            
            if ic > best_ic:
                best_ic = ic
                best_params = {"alpha": alpha, "l1_ratio": l1}

    print(f"Best Params (Val): {best_params} | Val Rank IC Mean: {best_ic:.6f}")

    # 5. Final Evaluation on Test Set (2024)
    print("Evaluating on 2024 Test Set with best hyperparameters...")
    final_model = ElasticNet(**best_params, random_state=42, max_iter=10000)
    final_model.fit(X_train, y_train)
    
    test_preds = final_model.predict(X_test)
    test_rank_ic = calculate_rank_ic(y_test, test_preds, test_df['report_year']) # Pass grouping variable
    test_mse = mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    n_nonzero = np.sum(final_model.coef_ != 0)

    # 6. Save Results
    test_df['predicted_return'] = test_preds
    test_df[['symbol', 'report_year', CONFIG["TARGET_COL"], 'predicted_return']].to_csv(CONFIG["PREDS_OUTPUT"], index=False)
    
    summary = {
        "test_year": CONFIG["TEST_YEAR"],
        "best_alpha": best_params["alpha"],
        "best_l1_ratio": best_params["l1_ratio"],
        "val_rank_ic_mean": best_ic,
        "test_rank_ic_mean": test_rank_ic,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "features_total": len(features_cols),
        "features_retained": int(n_nonzero)
    }
    with open(CONFIG["SUMMARY_OUTPUT"], 'w') as f:
        json.dump(summary, f, indent=4)

    # 7. Final Output
    print("="*50)
    print(f"FINAL TEST PERFORMANCE (YEAR: {CONFIG['TEST_YEAR']})")
    print("="*50)
    print(f"Rank IC Mean (Spearman): {test_rank_ic:.6f}")
    print(f"MSE                    : {test_mse:.6f}")
    print(f"MAE                    : {test_mae:.6f}")
    print("-" * 50)
    print(f"Model keeps {n_nonzero} / {len(features_cols)} features.")
    print(f"Full summary saved to {CONFIG['SUMMARY_OUTPUT']}")
    print("="*50)

if __name__ == "__main__":
    main()