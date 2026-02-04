import os
import re
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CONFIG = {
    "INPUT_CSV": DATA_DIR / "clean_master.csv",
    "TXT_DIR": DATA_DIR / "texts",
    "OUTPUT_DIR": DATA_DIR / "mda_texts",
    "LOG_FILE": PROJECT_ROOT / "extraction.log",
    "WORKERS": max(1, os.cpu_count() - 1),
    "MIN_CONTENT_LENGTH": 200  # Minimum characters to be considered valid MD&A
}

# Pre-compiled Regex Patterns for Performance
# Capture common variations of MD&A headers
START_PATTERNS = [
    re.compile(r"^第[三四]节\s*管理层讨论与分析"),
    re.compile(r"^董事会报告"),
    re.compile(r"^经营情况讨论与分析"),
    re.compile(r"^管理层讨论与分析")
]

# Capture common next sections to mark the end
END_PATTERNS = [
    re.compile(r"^第[四五六]节\s*重要事项"),
    re.compile(r"^第[四五六]节\s*公司治理"),
    re.compile(r"^第[五六]节\s*环境"),
    re.compile(r"^重要事项"),
    re.compile(r"^公司治理"),
    re.compile(r"^回顾与展望") # Sometimes appears in summary
]

# Regex to detect Table of Content lines (lines ending in digits)
TOC_PATTERN = re.compile(r"(\.{3,}|\s)\d+\s*$")

# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8')]
)

def is_likely_toc(line):
    """
    Check if a line looks like a Table of Contents entry.
    Strategy: Check if it ends with a number (page number) or dots.
    """
    if len(line) > 100: # TOC lines are rarely huge
        return False
    return bool(TOC_PATTERN.search(line))

def match_pattern(line, patterns):
    """Check if line matches any of the compiled regex patterns."""
    line = line.strip()
    # Optimization: Only check short lines (headers) to save CPU
    if len(line) > 50: 
        return False
    
    # Normalize spaces for matching
    clean_line = re.sub(r"\s+", "", line)
    
    for pattern in patterns:
        # Match against cleaned line or original depending on regex design
        # Here we match loose structure
        if pattern.search(line) or pattern.search(clean_line):
            return True
    return False

def extract_mda(row):
    try:
        symbol = str(row['symbol']).zfill(6)
        year = str(row['report_year'])
        txt_name = f"{symbol}_{year}.txt"
        out_name = f"{symbol}_{year}_mda.txt"
        
        input_path = CONFIG["TXT_DIR"] / year / txt_name
        output_dir = CONFIG["OUTPUT_DIR"] / year
        output_path = output_dir / out_name
        
        if not input_path.exists():
            return "missing_input"
            
        if output_path.exists() and output_path.stat().st_size > CONFIG["MIN_CONTENT_LENGTH"]:
            return "skipped"
            
        # Read content
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        start_idx = -1
        end_idx = -1
        
        # 1. Find Start Position (Skip TOC)
        for i, line in enumerate(lines):
            # Heuristic: TOC usually resides in the first 500 lines
            if i < 500 and is_likely_toc(line):
                continue
                
            if match_pattern(line, START_PATTERNS):
                start_idx = i
                break
                
        if start_idx == -1:
            return "header_not_found"
            
        # 2. Find End Position
        # Start searching 5 lines after start_idx to avoid immediate false stop
        for i in range(start_idx + 5, len(lines)):
            line = lines[i]
            if match_pattern(line, END_PATTERNS):
                end_idx = i
                break
                
        # 3. Validation & Extraction
        # If no end found, cap at 3000 lines from start (safety net)
        if end_idx == -1:
            end_idx = min(start_idx + 3000, len(lines))
            status = "forced_end"
        else:
            status = "success"
            
        content = "".join(lines[start_idx:end_idx])
        
        # Filter out empty or extremely short extractions
        if len(content) < CONFIG["MIN_CONTENT_LENGTH"]:
            return "too_short"

        # 4. Save
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return status

    except Exception as e:
        logging.error(f"Error {symbol}: {str(e)}")
        return "failed"

def main():
    print(f"Smart Parser (Regex Engine) | Workers: {CONFIG['WORKERS']}")
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    
    if not CONFIG["INPUT_CSV"].exists():
        logging.critical("Input CSV missing.")
        return

    df = pd.read_csv(CONFIG["INPUT_CSV"])
    stats = {
        "success": 0, 
        "forced_end": 0, 
        "header_not_found": 0, 
        "too_short": 0, 
        "skipped": 0,
        "missing_input": 0,
        "failed": 0
    }
    
    with ProcessPoolExecutor(max_workers=CONFIG["WORKERS"]) as executor:
        futures = {executor.submit(extract_mda, row): row['symbol'] for _, row in df.iterrows()}
        
        with tqdm(total=len(df), unit="file", desc="Extracting") as pbar:
            for future in as_completed(futures):
                res = future.result()
                stats[res] = stats.get(res, 0) + 1
                
                pbar.set_postfix(
                    OK=stats["success"] + stats["forced_end"], 
                    Miss=stats["header_not_found"]
                )
                pbar.update(1)

    print("\n" + "="*40)
    print("✅ EXTRACTION SUMMARY")
    print(f"Success (Clean)  : {stats['success']}")
    print(f"Success (Forced) : {stats['forced_end']} (No end header found, capped)")
    print(f"Skipped          : {stats['skipped']}")
    print(f"Not Found        : {stats['header_not_found']} (Title mismatch/OCR error)")
    print(f"Too Short        : {stats['too_short']} (Extraction failed sanity check)")
    print(f"Missing Input    : {stats['missing_input']}")
    print("-" * 40)
    print(f"Output Directory : {CONFIG['OUTPUT_DIR']}")
    print("="*40)

if __name__ == "__main__":
    main()