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
    "MIN_CONTENT_LENGTH": 500,  # Valid MD&A should be reasonably long
    "WARN_CONTENT_LENGTH": 2000 # Warn if content is surprisingly short for an annual report
}

# Pre-compiled Regex Patterns for Performance
START_PATTERNS = [
    re.compile(r"^第[三四]节\s*管理层讨论与分析"),
    re.compile(r"^第[三四]节\s*董事会报告"),
    re.compile(r"^经营情况讨论与分析"),
    re.compile(r"^管理层讨论与分析"),
    re.compile(r"^董事会报告"),
    re.compile(r"^业务回顾与展望") # Sometimes used as start in older reports
]

# Robust End Patterns - These are usually high-level section markers that follow MD&A
END_PATTERNS = [
    re.compile(r"^第[四五六七]节\s*重要事项"),
    re.compile(r"^第[四五六七]节\s*股份变动及股东情况"),
    re.compile(r"^第[四五六七]节\s*公司治理"),
    re.compile(r"^第[四五六七]节\s*环境与社会责任"),
    re.compile(r"^重要事项"),
    re.compile(r"^公司治理"),
    re.compile(r"^财务报表")
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
    """Check if a line looks like a Table of Contents entry."""
    if len(line) > 150: 
        return False
    return bool(TOC_PATTERN.search(line))

def match_pattern(line, patterns):
    """Check if line matches any of the compiled regex patterns."""
    line = line.strip()
    # Increased limit to 100 to catch long titles with brackets/suffixes
    if not line or len(line) > 100: 
        return False
    
    # Normalize spaces for matching
    clean_line = re.sub(r"\s+", "", line)
    
    for pattern in patterns:
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
            # Only match headers in the first 40% of the document to avoid false positives in appendices
            if i > len(lines) * 0.4:
                break
                
            # Heuristic: TOC usually resides in the first few hundred lines
            if i < 800 and is_likely_toc(line):
                continue
                
            if match_pattern(line, START_PATTERNS):
                start_idx = i
                break
                
        if start_idx == -1:
            return "header_not_found"
            
        # 2. Find End Position
        # Start searching 20 lines after start_idx to avoid immediate false stop
        # Search until near the end of the file
        for i in range(start_idx + 20, len(lines)):
            line = lines[i]
            if match_pattern(line, END_PATTERNS):
                end_idx = i
                break
                
        # 3. Validation & Extraction
        # If no end found, cap at 5000 lines from start (increased safety net)
        if end_idx == -1:
            end_idx = min(start_idx + 5000, len(lines))
            status = "forced_end"
        else:
            status = "success"
            
        content = "".join(lines[start_idx:end_idx])
        
        # Filter out empty or extremely short extractions
        if len(content) < CONFIG["MIN_CONTENT_LENGTH"]:
            return "too_short"
            
        if len(content) < CONFIG["WARN_CONTENT_LENGTH"]:
            logging.warning(f"Short extraction for {symbol}_{year}: {len(content)} chars. Check if truncated.")

        # 4. Save
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return status

    except Exception as e:
        logging.error(f"Error {symbol}: {str(e)}")
        return "failed"

def main():
    print(f"MD&A Extractor (Refined) | Workers: {CONFIG['WORKERS']}")
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

    print()
    print("="*40)
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