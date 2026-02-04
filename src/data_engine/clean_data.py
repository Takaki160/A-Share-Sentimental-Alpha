import os
import shutil
import logging
import pandas as pd
import pdfplumber
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ================= Configuration =================
# 自动获取项目根目录 (A-SHARE-SENTIMENTAL-ALPHA)
# 逻辑：当前脚本在 src/data_engine/，向上回退 2 级即为根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CONFIG = {
    # 路径配置 (全自动适配)
    "INPUT_CSV": DATA_DIR / "master_alignment_table.csv",
    "OUTPUT_CSV": DATA_DIR / "clean_master.csv",
    "REPORT_DIR": DATA_DIR / "reports",
    "TRASH_DIR": DATA_DIR / "trash",
    "LOG_FILE": PROJECT_ROOT / "cleaning.log",
    
    # 性能配置
    "WORKERS": max(1, os.cpu_count() - 2), # 自动留出 2 个核给系统
    
    # 业务阈值
    "THRESHOLDS": {
        "MIN_BYTES": 500 * 1024,       # < 500KB: 极大概率是垃圾
        "SAFE_BYTES": 2 * 1024 * 1024, # > 2MB: 极大概率是正文
        "CHECK_CHARS": 600             # 检查前 600 字符 (覆盖头部标题区)
    }
}
# ===============================================

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def setup_dirs():
    """Ensure output and trash directories exist."""
    CONFIG["TRASH_DIR"].mkdir(parents=True, exist_ok=True)
    if not CONFIG["REPORT_DIR"].exists():
        logging.error(f"Report directory not found: {CONFIG['REPORT_DIR']}")
        raise FileNotFoundError(f"Report directory missing: {CONFIG['REPORT_DIR']}")

def is_content_summary(filepath):
    """
    Opens PDF and checks first page text for summary keywords.
    Returns: (is_summary: bool, reason: str)
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            if not pdf.pages:
                return True, "Empty PDF"
            
            # 读取第一页
            first_page_text = pdf.pages[0].extract_text() or ""
            header_text = first_page_text[:CONFIG["THRESHOLDS"]["CHECK_CHARS"]]
            
            # 关键词过滤逻辑
            if "摘要" in header_text:
                # 二次确认：有时正文标题包含摘要，但这里我们主要杀标题里的
                return True, "Keyword '摘要' found in header"
            
            if "取消" in header_text and "公告" in header_text:
                return True, "Keyword '取消公告' found"
                
            return False, "Content OK"
            
    except Exception as e:
        # PDF 损坏或加密无法读取，视为无效
        return True, f"Read Error: {str(e)}"

def process_single_file(row_data):
    """
    Worker function to process a single row.
    """
    symbol = str(row_data['symbol']).zfill(6)
    year = str(row_data['report_year'])
    filename = f"{symbol}_{year}.pdf"
    
    # 构建绝对路径
    file_path = CONFIG["REPORT_DIR"] / year / filename
    trash_path = CONFIG["TRASH_DIR"] / year / filename
    
    result = {
        "row": row_data,
        "status": "valid",
        "reason": "",
        "file_name": filename
    }

    if not file_path.exists():
        result["status"] = "missing"
        return result

    file_size = file_path.stat().st_size

    # 1. 快速过滤：太小 (<500KB)
    if file_size < CONFIG["THRESHOLDS"]["MIN_BYTES"]:
        result["status"] = "invalid"
        result["reason"] = f"Size too small ({file_size/1024:.1f}KB)"
        _move_to_trash(file_path, trash_path)
        return result

    # 2. 快速通行：够大 (>2MB)
    if file_size > CONFIG["THRESHOLDS"]["SAFE_BYTES"]:
        result["status"] = "valid"
        return result

    # 3. 深度检查：灰度区域 (500KB - 2MB)
    is_summ, reason = is_content_summary(file_path)
    if is_summ:
        result["status"] = "invalid"
        result["reason"] = reason
        _move_to_trash(file_path, trash_path)
    else:
        result["status"] = "valid"

    return result

def _move_to_trash(src, dest):
    """Helper to safely move files to trash bin."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
    except Exception as e:
        logging.error(f"Failed to move {src.name} to trash: {e}")

def main():
    print(f"Project Root Detected: {PROJECT_ROOT}")
    setup_dirs()
    
    if not CONFIG["INPUT_CSV"].exists():
        logging.error(f"Input file not found: {CONFIG['INPUT_CSV']}")
        return

    df = pd.read_csv(CONFIG["INPUT_CSV"])
    logging.info(f"Starting cleanup process. Total records: {len(df)}")
    
    valid_rows = []
    stats = {"valid": 0, "missing": 0, "invalid": 0}
    
    # 启动多进程池
    with ProcessPoolExecutor(max_workers=CONFIG["WORKERS"]) as executor:
        futures = {executor.submit(process_single_file, row): row['symbol'] for _, row in df.iterrows()}
        
        # 进度条
        with tqdm(total=len(df), unit="file", desc="Cleaning") as pbar:
            for future in as_completed(futures):
                res = future.result()
                
                if res["status"] == "valid":
                    valid_rows.append(res["row"])
                    stats["valid"] += 1
                elif res["status"] == "missing":
                    stats["missing"] += 1
                    # 缺失文件可以选择记下来，但不作为错误抛出
                else: # invalid
                    stats["invalid"] += 1
                    logging.info(f"Removed: {res['file_name']} | Reason: {res['reason']}")
                
                pbar.update(1)

    # 保存清洗后的 CSV
    clean_df = pd.DataFrame(valid_rows)
    clean_df.to_csv(CONFIG["OUTPUT_CSV"], index=False)
    
    # 打印最终报告
    report = f"""
    ===========================================
    ✅ CLEANUP COMPLETED
    ===========================================
    Project Root     : {PROJECT_ROOT}
    Original Dataset : {len(df)}
    Cleaned Dataset  : {len(clean_df)}
    
    [Stats]
    - Valid (Kept)   : {stats['valid']}
    - Missing        : {stats['missing']}
    - Invalid (Trash): {stats['invalid']}
    
    [Output]
    - Clean CSV      : {CONFIG['OUTPUT_CSV']}
    - Trash Bin      : {CONFIG['TRASH_DIR']}
    ===========================================
    """
    print(report)
    logging.info("Cleanup process finished successfully.")

if __name__ == "__main__":
    main()