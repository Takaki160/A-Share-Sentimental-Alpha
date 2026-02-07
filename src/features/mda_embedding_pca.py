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
    
    # --- 模型选择 ---
    # 哈工大 RoBERTa-wwm-ext (中文金融领域表现优异)
    "MODEL_NAME": "hfl/chinese-roberta-wwm-ext",
    
    # --- 针对 RTX 3050 Laptop 的显存优化 ---
    "USE_FP16": True,          # 必须开启！显存占用减半
    "BATCH_SIZE": 8,           # 3050 Laptop 4GB 显存建议设为 4-8
    "MAX_LEN": 512,            # 模型硬限制
    
    # --- 关键策略: Head + Tail ---
    "HEAD_CHUNKS": 3,          # 取前 3 个切片
    "TAIL_CHUNKS": 3,          # 取后 3 个切片
    "CHUNK_OVERLAP": 50,       # 切片重叠长度，保持语义连贯
    
    # --- PCA 设置 ---
    "N_COMPONENTS": 80,        # 降维目标
    "RANDOM_STATE": 42,
    "PCA_FIT_YEAR_CUTOFF": 2022 # 严禁使用此年份之后的数据训练 PCA (防泄漏)
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
    """针对 RTX 显卡优先使用 CUDA"""
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
    工业级 Pooling: 对所有 Token 的向量取加权平均 (排除 Padding)
    """
    token_embeddings = model_output.last_hidden_state # [Batch, Seq, Hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask

def process_single_file(file_path, tokenizer, model, device):
    """
    读取单文件 -> Head+Tail 智能切片 -> 推理 -> Mean Pooling
    """
    try:
        # 1. 鲁棒的文件读取
        text = ""
        encodings = ['utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read().strip()
                break
            except UnicodeDecodeError:
                continue
                
        if not text or len(text) < 50: # 忽略过短的垃圾文件
            return None

        # 2. Tokenize 全文 (获取 input_ids 列表)
        # 注意：这里 truncation=False，我们要手动切片
        tokens = tokenizer(text, add_special_tokens=True, return_tensors='pt', verbose=False)
        input_ids_all = tokens['input_ids'][0] # 1D Tensor
        
        total_tokens = len(input_ids_all)
        
        # 3. 智能选择 Head + Tail 的切片索引
        chunk_size = CONFIG["MAX_LEN"]
        overlap = CONFIG["CHUNK_OVERLAP"]
        stride = chunk_size - overlap
        
        # 生成所有切片的起始位置
        all_starts = list(range(0, total_tokens, stride))
        
        selected_starts = []
        needed_chunks = CONFIG["HEAD_CHUNKS"] + CONFIG["TAIL_CHUNKS"]
        
        if len(all_starts) <= needed_chunks:
            # 如果文本不够长，就全部都要
            selected_starts = all_starts
        else:
            # 头部 N 个
            selected_starts.extend(all_starts[:CONFIG["HEAD_CHUNKS"]])
            # 尾部 M 个 (去重)
            tail_starts = all_starts[-CONFIG["TAIL_CHUNKS"]:]
            for s in tail_starts:
                if s not in selected_starts:
                    selected_starts.append(s)
        
        # 4. 构建 Batch Tensor
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
            
        # 堆叠成 Batch [Batch_Size, Seq_Len]
        # RTX 3050 显存优化: 如果 chunks 数量超过配置的 BATCH_SIZE，可以再分批，但这里 Head+Tail 通常只有 4 个，很安全
        input_ids_tensor = torch.stack(chunk_input_ids).to(device)
        att_mask_tensor = torch.stack(chunk_att_masks).to(device)
        
        # 5. 模型推理 (开启混合精度)
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=att_mask_tensor)
            chunk_embeddings = mean_pooling(outputs, att_mask_tensor) # [Chunks, Hidden]
        
        # 6. 聚合文档向量 (对所有切片取平均)
        doc_embedding = torch.mean(chunk_embeddings, dim=0).cpu().numpy()
        
        return doc_embedding

    except Exception as e:
        logging.warning(f"Error processing {file_path.name}: {str(e)}")
        return None

# ================= Main Pipeline =================

def main():
    logging.info("🚀 Starting Industrial MD&A Embedding Pipeline (RTX 3050 Optimized)")
    
    device = get_device()
    
    # 1. 加载模型
    logging.info(f"Loading Model: {CONFIG['MODEL_NAME']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    model = AutoModel.from_pretrained(CONFIG['MODEL_NAME'])
    model.to(device)
    model.eval()
    
    if CONFIG["USE_FP16"] and device.type == 'cuda':
        model.half()
        logging.info("⚡ FP16 Precision Enabled (Memory Usage Halved)")

    # 2. 扫描文件 (核心修改：使用 rglob 递归搜索子文件夹)
    # rglob = recursive glob，会查找所有子目录下的 txt
    files = list(CONFIG["INPUT_DIR"].rglob("*.txt"))
    
    if not files:
        logging.error(f"No .txt files found in {CONFIG['INPUT_DIR']} or its subfolders!")
        logging.error("Check if the path is correct: " + str(CONFIG['INPUT_DIR']))
        return
    logging.info(f"Found {len(files)} files to process.")

    # 3. 批量生成 Embeddings
    embeddings = []
    metadata = []
    
    # 正则1: 标准格式 (文件名包含股票和年份) e.g., "000001_2022.txt"
    pattern_standard = re.compile(r"(\d{6}).*?(\d{4})")
    # 正则2: 仅股票代码 (年份从文件夹取) e.g., "2022/000001.txt"
    pattern_symbol_only = re.compile(r"(\d{6})")
    
    pbar = tqdm(files, desc="Embedding Generation")
    for file_path in pbar:
        symbol = None
        year = None
        
        # --- 尝试解析元数据 (增强版) ---
        # 策略 A: 尝试从文件名直接读取 "代码+年份"
        match = pattern_standard.search(file_path.name)
        if match:
            symbol, year = match.groups()
            year = int(year)
        else:
            # 策略 B: 如果文件名没年份，尝试从 "父文件夹名" 读取年份
            # 适用于结构: .../2022/000001.txt
            match_sym = pattern_symbol_only.search(file_path.name)
            if match_sym:
                symbol = match_sym.group(1)
                # 检查父文件夹是否是数字 (年份)
                if file_path.parent.name.isdigit():
                    year = int(file_path.parent.name)
        
        # 如果依然无法解析出年份或代码，跳过
        if symbol is None or year is None:
            logging.warning(f"Skipping unparseable file: {file_path}")
            continue
            
        # 处理文件
        emb = process_single_file(file_path, tokenizer, model, device)
        
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

    # --- Step 4: PCA 降维 (严格防泄露) ---
    logging.info("-" * 40)
    logging.info(f"Step 4: PCA Fit (Training strictly on Data <= {CONFIG['PCA_FIT_YEAR_CUTOFF']})")
    
    train_mask = df_meta['report_year'] <= CONFIG['PCA_FIT_YEAR_CUTOFF']
    X_train = X_all[train_mask]
    
    n_train = len(X_train)
    logging.info(f"PCA Training Samples: {n_train} (Total: {len(X_all)})")
    
    # 放宽一点限制，防止样本略少报错 (只要大于 Components 数量即可)
    if n_train < CONFIG["N_COMPONENTS"]:
        logging.error(f"FATAL: PCA training samples ({n_train}) too small for {CONFIG['N_COMPONENTS']} components!")
        logging.error("Check if your data contains years <= 2022.")
        return

    pca = PCA(n_components=CONFIG["N_COMPONENTS"], random_state=CONFIG["RANDOM_STATE"])
    pca.fit(X_train)
    
    explained_ratio = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA Variance Explained: {explained_ratio:.2%}")

    # 转换全量数据
    X_pca = pca.transform(X_all)
    
    # --- Step 5: 保存结果 ---
    pca_cols = [f"pca_{i+1}" for i in range(CONFIG["N_COMPONENTS"])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    # 合并 Meta 和 PCA 特征
    df_final = pd.concat([df_meta.reset_index(drop=True), df_pca], axis=1)
    
    df_final.to_csv(CONFIG["OUTPUT_CSV"], index=False)
    logging.info(f"✅ Features saved to {CONFIG['OUTPUT_CSV']}")

if __name__ == "__main__":
    main()