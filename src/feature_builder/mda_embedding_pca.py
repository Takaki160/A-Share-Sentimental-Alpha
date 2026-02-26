import os
import sys
import re
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT
LOG_DIR.mkdir(exist_ok=True)

CONFIG = {
    "INPUT_DIR": DATA_DIR / "mda_texts",
    "OUTPUT_CSV": DATA_DIR / "features_pca_embeddings.csv",
    "LOG_FILE": LOG_DIR / "feature_engineering.log",
    
    # Model Configuration
    "MODEL_NAME": "hfl/chinese-roberta-wwm-ext", # Chinese RoBERTa-wwm-ext (Whole Word Masking)
    
    # Optimization for RTX 3050
    "USE_FP16": True,
    "BATCH_SIZE": 8,
    "MAX_LEN": 512,
    
    # Chunking Strategy
    "HEAD_CHUNKS": 3,
    "TAIL_CHUNKS": 3,
    "CHUNK_OVERLAP": 50,
    
    # PCA Configuration
    "N_COMPONENTS": 80,
    "RANDOM_STATE": 42,
    "PCA_FIT_YEAR_CUTOFF": 2023 # Only use data from 2023 and earlier to prevent future data leakage
}

# ================= Setup Logging =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ================= Helper Functions =================
def get_device():
    """Detect if GPU is available and return appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        logging.info(f"GPU Detected: {props.name} | VRAM: {props.total_memory / 1024**3:.2f} GB")
        return device
    else:
        logging.warning("No GPU detected! Running on CPU will be very slow.")
        return torch.device("cpu")

def mean_pooling(model_output, attention_mask):
    """
    Pooling strategy to get fixed-size sentence embeddings.
    We take the mean of the token embeddings, weighted by the attention mask to ignore padding tokens
    """
    token_embeddings = model_output.last_hidden_state # [Batch, Seq, Hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask

def process_single_file(file_path, tokenizer, model, device):
    """
    Process a single MD&A text file to generate its embedding.
    Steps:
    1. Read the text file with robust encoding handling.
    2. Tokenize the entire text without truncation.
    3. Select intelligent chunks from the head and tail of the document.
    4. Create batch tensors for the selected chunks.
    5. Run the model in inference mode to get chunk embeddings.
    6. Aggregate chunk embeddings into a single document embedding.
    """
    try:
        # 1. Read the text file with robust encoding handling (try utf-8, then fallback to gbk)
        text = ""
        encodings = ['utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read().strip()
                break
            except UnicodeDecodeError:
                continue
                
        if not text or len(text) < 50: # If text is too short, likely a read error or empty file
            return None

        # 2. Tokenize the entire text without truncation (we will handle chunking manually)
        tokens = tokenizer(text, add_special_tokens=True, return_tensors='pt', verbose=False)
        input_ids_all = tokens['input_ids'][0] # 1D Tensor
        
        total_tokens = len(input_ids_all)
        
        # 3. Select intelligent chunks from the head and tail of the document
        chunk_size = CONFIG["MAX_LEN"]
        overlap = CONFIG["CHUNK_OVERLAP"]
        stride = chunk_size - overlap
        
        all_starts = list(range(0, total_tokens, stride)) # Calculate chunk start indices
        
        selected_starts = []
        needed_chunks = CONFIG["HEAD_CHUNKS"] + CONFIG["TAIL_CHUNKS"]
        
        if len(all_starts) <= needed_chunks:
            selected_starts = all_starts # If document is very short, just take all chunks (with overlap) without strict head/tail separation
        else:
            selected_starts.extend(all_starts[:CONFIG["HEAD_CHUNKS"]]) # First M chunks from the head
            tail_starts = all_starts[-CONFIG["TAIL_CHUNKS"]:] # Last M chunks from the tail
            for s in tail_starts:
                if s not in selected_starts:
                    selected_starts.append(s)
        
        # 4. Create batch tensors for the selected chunks (with padding if necessary)
        chunk_input_ids = []
        chunk_att_masks = []
        
        for start in selected_starts:
            end = min(start + chunk_size, total_tokens)
            ids = input_ids_all[start:end]
            
            # Padding
            padding_len = chunk_size - len(ids)
            if padding_len > 0:
                ids = torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)])
                mask = torch.cat([torch.ones(len(ids)-padding_len), torch.zeros(padding_len)])
            else:
                mask = torch.ones(len(ids))
                
            chunk_input_ids.append(ids)
            chunk_att_masks.append(mask)
            
        if not chunk_input_ids:
            return None

        input_ids_tensor = torch.stack(chunk_input_ids).to(device)
        att_mask_tensor = torch.stack(chunk_att_masks).to(device)
        
        # 5. Run the model in inference mode to get chunk embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=att_mask_tensor)
            chunk_embeddings = mean_pooling(outputs, att_mask_tensor) # [Chunks, Hidden]
        
        # 6. Aggregate chunk embeddings into a single document embedding (mean pooling across chunks)
        doc_embedding = torch.mean(chunk_embeddings, dim=0).cpu().numpy()
        
        return doc_embedding

    except Exception as e:
        logging.warning(f"Error processing {file_path.name}: {str(e)}")
        return None

# ================= Main Pipeline =================
def main():
    device = get_device()
    logging.info(f"🚀 Starting MD&A Embedding Pipeline on device {device}")
    
    # 1. Load the pre-trained model and tokenizer
    logging.info(f"Loading Model: {CONFIG['MODEL_NAME']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    model = AutoModel.from_pretrained(CONFIG['MODEL_NAME'])
    model.to(device)
    model.eval()
    
    if CONFIG["USE_FP16"] and device.type == 'cuda':
        model.half()
        logging.info("⚡ FP16 Precision Enabled (Memory Usage Halved)")

    # 2. Find all text files in the input directory (including subfolders)
    files = list(CONFIG["INPUT_DIR"].rglob("*.txt")) # Recursively find all .txt files
    
    if not files:
        logging.error(f"No .txt files found in {CONFIG['INPUT_DIR']} or its subfolders!")
        logging.error("Check if the path is correct: " + str(CONFIG['INPUT_DIR']))
        return
    logging.info(f"Found {len(files)} files to process.")

    # 3. Process each file to generate embeddings, while collecting metadata
    embeddings = []
    metadata = []
    
    pattern_standard = re.compile(r"(\d{6}).*?(\d{4})") # Matches "000001_2022.txt" or "2022_000001.txt" and captures code and year
    pattern_symbol_only = re.compile(r"(\d{6})") # Matches "000001.txt" and captures code only (year will be inferred from parent folder)
    
    pbar = tqdm(files, desc="Embedding Generation")
    for file_path in pbar:
        symbol = None
        year = None
        
        match = pattern_standard.search(file_path.name)
        if match: # First try to extract symbol and year from filename
            symbol, year = match.groups()
            year = int(year)
        else: # If that fails, try to extract symbol only and infer year from parent folder
            match_sym = pattern_symbol_only.search(file_path.name)
            if match_sym:
                symbol = match_sym.group(1)
                if file_path.parent.name.isdigit():
                    year = int(file_path.parent.name)
        
        if symbol is None or year is None:
            logging.warning(f"Skipping unparseable file: {file_path}")
            continue
            
        emb = process_single_file(file_path, tokenizer, model, device) # Generate embedding for this file
        
        if emb is not None:
            embeddings.append(emb)
            metadata.append({
                "symbol": symbol,
                "report_year": year,
                "file_name": file_path.name
            })
            
    if not embeddings:
        logging.error("No embeddings generated. Exiting.")
        return

    X_all = np.vstack(embeddings)
    df_meta = pd.DataFrame(metadata)
    logging.info(f"Embedding Generation Complete. Shape: {X_all.shape}")

    # 4. Fit PCA on the training set (reports from 2023 and earlier) and transform all data
    logging.info("-" * 40)
    logging.info(f"Step 4: PCA Fit (Training strictly on Data <= {CONFIG['PCA_FIT_YEAR_CUTOFF']})")
    
    train_mask = df_meta['report_year'] <= CONFIG['PCA_FIT_YEAR_CUTOFF']
    X_train = X_all[train_mask]
    
    n_train = len(X_train)
    logging.info(f"PCA Training Samples: {n_train} (Total: {len(X_all)})")
    
    if n_train < CONFIG["N_COMPONENTS"]: # PCA components cannot exceed number of samples
        logging.error(f"FATAL: PCA training samples ({n_train}) too small for {CONFIG['N_COMPONENTS']} components!")
        logging.error("Check if your data contains years <= 2022.")
        return

    pca = PCA(n_components=CONFIG["N_COMPONENTS"], random_state=CONFIG["RANDOM_STATE"])
    pca.fit(X_train)
    
    explained_ratio = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA Variance Explained: {explained_ratio:.2%}")

    X_pca = pca.transform(X_all) # Transform all data (including future years) to prevent data leakage in training phase, but PCA is only fit on past data
    
    # 5. Save the final features (PCA components) along with metadata to a CSV file
    pca_cols = [f"pca_{i+1}" for i in range(CONFIG["N_COMPONENTS"])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    df_final = pd.concat([df_meta.reset_index(drop=True), df_pca], axis=1) # Combine metadata and PCA features into one DataFrame
    
    df_final.to_csv(CONFIG["OUTPUT_CSV"], index=False)
    logging.info(f"✅ Features saved to {CONFIG['OUTPUT_CSV']}")

if __name__ == "__main__":
    main()