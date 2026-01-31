import os
import time
import random
import akshare as ak
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", message=".*fillna with 'method' is deprecated.*")

# ================= Configuration =================
START_YEAR = 2021
END_YEAR = 2024
HOLDING_PERIOD = 60         # T+60 trading days
MAX_WORKERS = 2             # Maintain low concurrency to avoid IP ban
OUTPUT_DIR = "data"
FINAL_FILE = os.path.join(OUTPUT_DIR, "master_alignment_table.csv")
# ===============================================

def safe_sleep(min_sec=0.1, max_sec=1.0):
    """Randomized sleep to respect API rate limits."""
    time.sleep(random.uniform(min_sec, max_sec))

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_master_schedule(report_period):
    """Fetch disclosure schedule and clean dates."""
    print(f"Fetching schedule for: {report_period}...")
    try:
        df = ak.stock_report_disclosure(market="沪深京", period=report_period)
        if df.empty: return pd.DataFrame()
        
        # Data Cleaning: Ensure valid dates and remove future disclosures
        df = df.dropna(subset=['实际披露'])
        df['实际披露'] = pd.to_datetime(df['实际披露'], errors='coerce').dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['实际披露'])
        
        today = datetime.now().strftime('%Y-%m-%d')
        df = df[df['实际披露'] <= today]
        
        return df[['股票代码', '股票简称', '实际披露']]
    except Exception as e:
        print(f"[Schedule Error]: {e}")
        return pd.DataFrame()

def get_pdf_url(symbol, disclosure_date):
    """Fetch PDF URL with strict keyword filtering and cleaning."""
    try:
        date_obj = datetime.strptime(disclosure_date, "%Y-%m-%d")
        start_str = date_obj.strftime("%Y%m%d")
        end_str = (date_obj + timedelta(days=3)).strftime("%Y%m%d") # Allow T+3 days window for report indexing
        
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol, market="沪深京", category="年报", 
            start_date=start_str, end_date=end_str
        )
        
        if df is None or df.empty: return None

        # Robust Cleaning: Handle non-string types and whitespace
        df.columns = df.columns.str.strip()
        if '公告标题' not in df.columns: return None
        df['公告标题'] = df['公告标题'].astype(str).str.strip()
        
        # Logic: (Contains "Annual Report" keywords) AND (Excludes "Summary/Cancel/English")
        mask_include = df['公告标题'].str.contains('年度报告', na=False) | \
                       df['公告标题'].str.contains('年报', na=False)
                       
        mask_exclude = ~df['公告标题'].str.contains('摘要', na=False) & \
                       ~df['公告标题'].str.contains('取消', na=False) & \
                       ~df['公告标题'].str.contains('英文', na=False) & \
                       ~df['公告标题'].str.contains('提示', na=False) & \
                       ~df['公告标题'].str.contains('更正', na=False) & \
                       ~df['公告标题'].str.contains('修订', na=False) & \
                       ~df['公告标题'].str.contains('更新', na=False)
        
        filtered_df = df[mask_include & mask_exclude].copy()
        
        if filtered_df.empty: return None
        
        # Heuristic: Shortest title is usually the main report (avoids "Updated", "Revised")
        filtered_df.loc[:, 'title_len'] = filtered_df['公告标题'].apply(len)
        best_match = filtered_df.sort_values('title_len').iloc[0]
        
        return {
            "title": best_match['公告标题'],
            "url": best_match['公告链接']
        }
    except Exception as e:
        print(f"[PDF Error] {symbol}: {e}")
        return None

def normalize_stock_code(code):
    code = str(code).strip()

    if code.startswith('6'):
        return f"{code}.SH"

    if code.startswith('0'):
        return f"{code}.SZ"

    return None # Only handle SZ and SH markets

# ts.set_token('YOUR_TUSHARE_TOKEN')

def calculate_label(symbol, disclosure_date):
    """Calculate T+1 buy and T+60 sell returns."""
    symbol = normalize_stock_code(symbol)
    if not symbol: return None

    try:
        start_dt = datetime.strptime(disclosure_date, "%Y-%m-%d")       
        fetch_start = start_dt.strftime("%Y%m%d")
        fetch_end = (start_dt + timedelta(days=150)).strftime("%Y%m%d") # Fetch wide range to ensure enough trading days
        
        # Adjust='hfq' is critical for long-term return calculation
        df_hist = ts.pro_bar(
            ts_code=symbol, adj='hfq', start_date=fetch_start, end_date=fetch_end
        )
        
        if df_hist is None or df_hist.empty: return None
            
        df_hist['trade_date'] = pd.to_datetime(df_hist['trade_date'])
        df_hist = df_hist.sort_values('trade_date').reset_index(drop=True)
        
        # Identify T+1 (Buy Date)
        mask_after = df_hist['trade_date'] > start_dt
        future_df = df_hist[mask_after].sort_values('trade_date')
        
        if future_df.empty: return None
            
        buy_row = future_df.iloc[0]
        buy_price = float(buy_row['open'])
        buy_date = buy_row['trade_date'].strftime("%Y-%m-%d")
        
        # Identify T+60 (Sell Date)
        if len(future_df) <= HOLDING_PERIOD: return None
            
        sell_row = future_df.iloc[HOLDING_PERIOD]
        sell_price = float(sell_row['close'])
        sell_date = sell_row['trade_date'].strftime("%Y-%m-%d")
        
        ret = (sell_price - buy_price) / buy_price
        
        return {
            "buy_date": buy_date, 
            "sell_date": sell_date,
            "return": ret
        }
    except Exception as e:
        print(f"[Label Error] {symbol}: {e}")
        return None

def process_single_stock(row, year):
    symbol = row['股票代码']
    disclosure_date = row['实际披露']
    
    safe_sleep()
    
    # 1. Get PDF
    pdf_info = get_pdf_url(symbol, disclosure_date)
    if not pdf_info: return None
        
    # 2. Get Market Data
    label_data = calculate_label(symbol, disclosure_date)
    if not label_data: return None

    print(f"✅ Success: {symbol}")

    return {
        "symbol": symbol,
        "name": row['股票简称'],
        "report_year": year,
        "disclosure_date": disclosure_date,
        "pdf_title": pdf_info['title'],
        "pdf_url": pdf_info['url'],
        "buy_date": label_data['buy_date'],
        "sell_date": label_data['sell_date'],
        "future_return": round(label_data['return'], 6)
    }

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Data Aligner Started | Years: {START_YEAR}-{END_YEAR} | Workers: {MAX_WORKERS}")
    
    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        schedule_df = get_master_schedule(f"{year}年报")
        if schedule_df.empty: continue
        
        print(f"Processing {year}: {len(schedule_df)} companies.")
        
        year_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_stock = {executor.submit(process_single_stock, row, year): row['股票代码'] for _, row in schedule_df.iterrows()}
            
            count = 0
            total = len(schedule_df)
            
            for future in as_completed(future_to_stock):
                count += 1
                try:
                    res = future.result()
                    if res:
                        year_results.append(res)
                        if len(year_results) % 20 == 0:
                            print(f"[{year}] Collected: {len(year_results)} | Last: {res['symbol']} ({res['future_return']:.2%})")
                except Exception:
                    pass
                    
                if count % 100 == 0:
                    print(f"Progress: {count}/{total}")

        # Save checkpoint
        if year_results:
            df_year = pd.DataFrame(year_results)
            checkpoint_path = os.path.join(OUTPUT_DIR, f"alignment_{year}.csv")
            df_year.to_csv(checkpoint_path, index=False)
            all_data.extend(year_results)
            print(f"Saved checkpoint for {year}: {len(df_year)} records.")
        else:
            print(f"No valid data for {year}.")

    # Save Master Table
    if all_data:
        final_df = pd.DataFrame(all_data)
        final_df.to_csv(FINAL_FILE, index=False)
        print(f"Workflow Complete. Master table saved to: {FINAL_FILE}")
        print(f"Total Records: {len(final_df)}")
    else:
        print("Workflow Failed: No data collected.")

if __name__ == "__main__":
    main()