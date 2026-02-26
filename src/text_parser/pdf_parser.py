import os
import re
import fitz  # PyMuPDF
import logging
import shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CONFIG = {
    "INPUT_CSV": DATA_DIR / "clean_master.csv",
    "PDF_DIR": DATA_DIR / "reports",
    "TXT_DIR": DATA_DIR / "texts",
    "BROKEN_DIR": DATA_DIR / "trash" / "broken_pdfs", # 新增：存放坏文件的地方
    "LOG_FILE": PROJECT_ROOT / "parsing.log",
    
    "WORKERS": max(1, os.cpu_count() - 2),
    "MIN_CONTENT_LENGTH": 500,
    "MIN_LINE_LENGTH": 4,
}
# ===============================================

# 1. 屏蔽 PyMuPDF 的底层 C++ 报错输出 (关键！)
# 这样终端就不会被 "xref error" 刷屏了
fitz.TOOLS.mupdf_display_errors(False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8')]
)

def clean_text_line(text: str) -> str:
    if not text: return ""
    # 去除不可见字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < CONFIG["MIN_LINE_LENGTH"] or line.isdigit():
            continue
        if "年度报告" in line and len(line) < 30:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def process_single_pdf(row) -> str:
    try:
        symbol = str(row['symbol']).zfill(6)
        year = str(row['report_year'])
        pdf_name = f"{symbol}_{year}.pdf"
        txt_name = f"{symbol}_{year}.txt"
        
        pdf_path = CONFIG["PDF_DIR"] / year / pdf_name
        txt_dir = CONFIG["TXT_DIR"] / year
        txt_path = txt_dir / txt_name
        
        if not pdf_path.exists():
            return "missing"
            
        if txt_path.exists() and txt_path.stat().st_size > CONFIG["MIN_CONTENT_LENGTH"]:
            return "skipped"
            
        txt_dir.mkdir(parents=True, exist_ok=True)

        full_text_list = []
        
        # ==========================================
        # 核心修改：增加对坏文件的防御
        # ==========================================
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    # 如果 PDF 损坏，这里通常会抛出 RuntimeError
                    text = page.get_text("text", sort=True)
                    cleaned = clean_text_line(text)
                    if cleaned:
                        full_text_list.append(cleaned)
                        
        except Exception as e:
            # 捕获到 PDF 损坏！
            logging.warning(f"Corrupted PDF detected: {pdf_name} - {str(e)}")
            
            # 策略：移动到坏文件目录，方便后续检查或删除
            CONFIG["BROKEN_DIR"].mkdir(parents=True, exist_ok=True)
            broken_path = CONFIG["BROKEN_DIR"] / pdf_name
            try:
                # 必须先关闭文件句柄（with 语句已自动处理）
                shutil.move(pdf_path, broken_path)
            except:
                pass # 如果移动失败（比如文件占用），暂时忽略
                
            return "corrupted" # 返回特殊状态
            
        # ==========================================

        final_content = "\n".join(full_text_list)
        
        if len(final_content) < CONFIG["MIN_CONTENT_LENGTH"]:
            return "too_short"

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        return "success"

    except Exception as e:
        logging.error(f"System Error {row.get('symbol')}: {e}")
        return "failed"

def main():
    print(f"PDF Parser (Self-Healing Mode) | Workers: {CONFIG['WORKERS']}")
    CONFIG["TXT_DIR"].mkdir(parents=True, exist_ok=True)
    
    if not CONFIG["INPUT_CSV"].exists(): return

    df = pd.read_csv(CONFIG["INPUT_CSV"])
    
    stats = {
        "success": 0, "skipped": 0, "failed": 0, 
        "missing": 0, "too_short": 0, "corrupted": 0
    }
    
    with ProcessPoolExecutor(max_workers=CONFIG["WORKERS"]) as executor:
        futures = {executor.submit(process_single_pdf, row): row['symbol'] for _, row in df.iterrows()}
        
        with tqdm(total=len(df), unit="file", desc="Parsing") as pbar:
            for future in as_completed(futures):
                status = future.result()
                stats[status] = stats.get(status, 0) + 1
                
                # 在进度条上实时显示坏文件数量
                pbar.set_postfix(
                    ok=stats["success"], 
                    skip=stats["skipped"], 
                    bad=stats["corrupted"] # 这里会显示坏文件数
                )
                pbar.update(1)

    print("\n" + "="*40)
    print("✅ PARSING COMPLETED")
    print(f"Success    : {stats['success']}")
    print(f"Skipped    : {stats['skipped']}")
    print(f"Corrupted  : {stats['corrupted']} (Moved to trash/broken_pdfs)")
    print(f"Too Short  : {stats['too_short']}")
    print(f"Failed     : {stats['failed']}")
    print("="*40)

if __name__ == "__main__":
    main()