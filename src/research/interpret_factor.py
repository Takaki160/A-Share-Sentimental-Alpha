# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_FILE = DATA_DIR / "features_pca_embeddings.csv"
LABEL_FILE = DATA_DIR / "master_alignment_table.csv"

def analyze_factor_meaning(factor_name, df, top_n=10):
    """分析某个因子得分最高和最低的样本"""
    print(f"\n{'='*20} 🔍 深度解析因子: {factor_name} {'='*20}")
    
    # 排序
    df_sorted = df.sort_values(by=factor_name, ascending=False)
    
    # 取最高分样本 (Top Winners)
    print(f"\n📈 [{factor_name}] 得分最高的公司 (该语义最强):")
    top_companies = df_sorted.head(top_n)
    print(top_companies[['symbol', 'report_year', factor_name]].to_string(index=False))
    
    # 取最低分样本 (Top Losers)
    print(f"\n📉 [{factor_name}] 得分最低的公司 (该语义最弱/相反):")
    bottom_companies = df_sorted.tail(top_n)
    print(bottom_companies[['symbol', 'report_year', factor_name]].to_string(index=False))

def main():
    # 1. 加载数据
    if not FEATURE_FILE.exists():
        print("特征文件不存在")
        return
        
    df = pd.read_csv(FEATURE_FILE, dtype={'symbol': str})
    
    # 2. 我们重点分析那个 IC=0.04 的神级因子，以及那个负向因子
    target_factors = ['pca_5', 'pca_8'] 
    
    for factor in target_factors:
        analyze_factor_meaning(factor, df)

if __name__ == "__main__":
    main()