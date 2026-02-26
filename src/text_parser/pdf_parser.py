import os
import re
import fitz  # PyMuPDF
fitz.TOOLS.mupdf_display_errors(False) # Suppress MuPDF warnings about broken PDFs
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
    "BROKEN_DIR": DATA_DIR / "trash" / "broken_pdfs",
    "LOG_FILE": PROJECT_ROOT / "parsing.log",
    
    "WORKERS": max(1, os.cpu_count() - 2),
    "MIN_CONTENT_LENGTH": 500,
    "MIN_LINE_LENGTH": 4,
}
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8')]
)

def clean_text_line(text: str) -> str:
    """Basic cleaning of extracted text lines:"""
    if not text: return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text) # Remove non-printable characters
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
    """Process a single PDF: extract text, clean it, and save to TXT. Returns status string."""
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
        
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text = page.get_text("text", sort=True)
                    cleaned = clean_text_line(text)
                    if cleaned:
                        full_text_list.append(cleaned)
                        
        except Exception as e:
            # MuPDF can throw various exceptions for corrupted PDFs, we catch them all to prevent crashing the whole process
            logging.warning(f"Corrupted PDF detected: {pdf_name} - {str(e)}")
            
            # Move the corrupted PDF to a separate folder for manual review
            CONFIG["BROKEN_DIR"].mkdir(parents=True, exist_ok=True)
            broken_path = CONFIG["BROKEN_DIR"] / pdf_name
            try:
                shutil.move(pdf_path, broken_path) # Move the corrupted PDF to the broken_pdfs folder
            except:
                pass
                
            return "corrupted"

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
                
                pbar.set_postfix(
                    ok=stats["success"], 
                    skip=stats["skipped"], 
                    bad=stats["corrupted"] # Show corrupted count in progress bar for quick monitoring
                )
                pbar.update(1)

    print()
    print("="*40)
    print("✅ PARSING COMPLETED")
    print(f"Success    : {stats['success']}")
    print(f"Skipped    : {stats['skipped']}")
    print(f"Corrupted  : {stats['corrupted']} (Moved to trash/broken_pdfs)")
    print(f"Too Short  : {stats['too_short']}")
    print(f"Failed     : {stats['failed']}")
    print("="*40)

if __name__ == "__main__":
    main()