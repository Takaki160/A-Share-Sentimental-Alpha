# -*- coding: utf-8 -*-
import os
import time
import random
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= Configuration =================
INPUT_FILE = "data/master_alignment_table.csv"
OUTPUT_DIR = "data/reports"
MAX_WORKERS = 8             # Concurrency level
MAX_RETRIES = 3             # Retry attempts per file
TIMEOUT = 15                # Request timeout in seconds
# ===============================================

def get_headers():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    return {"User-Agent": random.choice(agents)}

def download_single_pdf(row):
    """Downloads a single PDF with retry logic."""
    try:
        symbol = str(row['symbol']).zfill(6)
        year = str(row['report_year'])
        url = row['pdf_url']
        
        # Structure: data/reports/2022/000001_2022.pdf
        year_dir = os.path.join(OUTPUT_DIR, year)
        if not os.path.exists(year_dir):
            os.makedirs(year_dir, exist_ok=True)
            
        filename = f"{symbol}_{year}.pdf"
        filepath = os.path.join(year_dir, filename)
        
        # 1. Check if file exists (Resume capability)
        if os.path.exists(filepath):
            # Optional: Check file size > 1KB to ensure it's not a corrupted empty file
            if os.path.getsize(filepath) > 1024:
                return "skipped"
        
        # 2. Download with Retries
        for attempt in range(MAX_RETRIES):
            try:
                # Random sleep to prevent IP blocking
                time.sleep(random.uniform(0.1, 0.5))
                
                resp = requests.get(url, headers=get_headers(), timeout=TIMEOUT, stream=True)
                
                if resp.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return "success"
                elif resp.status_code == 404:
                    return "not_found"
                else:
                    time.sleep(1) # Wait before retry
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue
                
        return "failed"
        
    except Exception:
        return "error"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    print(f"PDF Downloader Started | Workers: {MAX_WORKERS}")
    
    df = pd.read_csv(INPUT_FILE)
    total_tasks = len(df)
    print(f"Found {total_tasks} documents to process.")

    stats = {"success": 0, "skipped": 0, "failed": 0, "not_found": 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(download_single_pdf, row): row['symbol'] for _, row in df.iterrows()}
        
        # Process with Progress Bar
        with tqdm(total=total_tasks, unit="file") as pbar:
            for future in as_completed(futures):
                status = future.result()
                
                if status == "success":
                    stats["success"] += 1
                    pbar.set_postfix_str(f"✅ {stats['success']} | ⏭️ {stats['skipped']}")
                elif status == "skipped":
                    stats["skipped"] += 1
                elif status == "not_found":
                    stats["not_found"] += 1
                else:
                    stats["failed"] += 1
                
                pbar.update(1)

    print("Download Complete!")
    print(f"Stats: Success: {stats['success']} | Skipped: {stats['skipped']} | Failed: {stats['failed']} | 404: {stats['not_found']}")
    print(f"Files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()