import os
import sys
import torch
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CONFIG = {
    "INPUT_DIR": DATA_DIR / "mda_texts",
    "OUTPUT_CSV": DATA_DIR / "features_pca_embeddings.csv",
    "LOG_FILE": PROJECT_ROOT / "feature_engineering.log",
    
    # 模型：UER RoBERTa (针对京东评论微调版，对中文褒贬义极其敏感)
    "MODEL_NAME": "uer/roberta-base-finetuned-jd-binary-chinese",
    
    # 硬件优化 (RTX 3050 4GB 专用配置)
    "BATCH_SIZE": 16,  # 显存安全水位
    "MAX_LEN": 512,
    "USE_FP16": True,
    
    # 降维目标
    "N_COMPONENTS": 20,
    "RANDOM_STATE": 42
}

# 自动设置 HF 镜像，防止国内下载模型超时
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class FeaturePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_system_status()
        
        logging.info(f"Loading backbone: {CONFIG['MODEL_NAME']}")
        # 使用 AutoModel 提取原始语义向量 (不带分类头)
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        self.model = AutoModel.from_pretrained(CONFIG['MODEL_NAME'])
        self.model.to(self.device)
        self.model.eval()

    def _log_system_status(self):
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            logging.info(f"Hardware: {props.name} | VRAM: {props.total_memory / 1024**3:.2f} GB")
            # Ampere 架构优化 (RTX 30系列)
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
        else:
            logging.warning("Running on CPU. Performance will be degraded.")

    def preprocess(self, text):
        """
        工业级预处理 (针对 2023/2024 年报样本深度优化)
        """
        if not text:
            return ""
        
        # --- 1. 深度清洗 (Deep Cleaning) ---
        
        # [安全修复] 使用字符串替换去除反斜杠(PDF乱码)，避免正则转义错误
        text = text.replace('\\', '') 
        
        # 去除 PDF 解析残留的 标签
        text = re.sub(r'<.*?>', '', text)
        
        # 去除 URL 链接 (防止 http://... 干扰语义)
        text = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)
        
        # 压缩多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # --- 2. 头部去噪 (Head Stripping) ---
        # 去除无意义的章节标题
        text = re.sub(r'第三节\s*管理层讨论与分析', '', text)
        text = re.sub(r'^[一二三四五12345]、\s*报告期内公司所处行业情况', '', text)
        
        # 去除交易所合规指引套话 (非贪婪匹配)
        # 样本: "公司需遵守《深圳证券交易所...指引》..."
        text = re.sub(r'公司需遵守.*?((指引)|(披露要求)|(规定))', '', text)
        
        # --- 3. 尾部切割 (Tail Surgery) ---
        # 目标：精准切除文末的"接待调研"及"备查文件"，保留紧邻其前的"风险/展望"
        noise_patterns = [
            r'(十[一二三四五六七八九]?|\d+)[、\. ]*报告期内?接待', # 匹配 "十二、报告期内接待..."
            r'接待调研、沟通、采访',
            r'备查文件目录',
            r'公司控制的结构化主体情况'
        ]
        
        trunc_idx = len(text)
        search_start = int(len(text) * 0.6) # 仅在后 40% 区域搜索，防止误伤正文
        
        for pattern in noise_patterns:
            match = re.search(pattern, text[search_start:])
            if match:
                abs_pos = search_start + match.start()
                # 找到最早出现的垃圾章节，截断位置设为该处
                if abs_pos < trunc_idx:
                    trunc_idx = abs_pos
        
        cleaned_body = text[:trunc_idx].strip()
        
        # --- 4. 长度适配 (Head-Tail Strategy) ---
        if len(cleaned_body) <= CONFIG["MAX_LEN"]:
            return cleaned_body
            
        # 超过 512 token 时，取头 + 尾
        # 此时的"尾"已经是被清洗过的高价值"风险/展望"部分
        limit = (CONFIG["MAX_LEN"] - 2) // 2
        return cleaned_body[:limit] + cleaned_body[-limit:]

    def extract_embeddings(self, texts):
        """
        提取 [CLS] Token 向量
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=CONFIG["MAX_LEN"], 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # 混合精度上下文管理器
            if CONFIG["USE_FP16"] and self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            # 提取 [CLS] (Index 0)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings.cpu().numpy()

def get_file_list():
    tasks = []
    if not CONFIG["INPUT_DIR"].exists():
        logging.error(f"Input directory missing: {CONFIG['INPUT_DIR']}")
        return tasks
        
    for year_dir in sorted(CONFIG["INPUT_DIR"].iterdir()):
        if year_dir.is_dir():
            for f in year_dir.glob("*_mda.txt"):
                try:
                    parts = f.name.split('_')
                    if len(parts) >= 2:
                        tasks.append({
                            "symbol": parts[0],
                            "report_year": int(parts[1]),
                            "path": f
                        })
                except Exception:
                    continue
    return tasks

def batch_generator(data, batch_size):
    """内存友好的批次生成器"""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def main():
    logging.info("Starting Feature Engineering Pipeline (Embedding + PCA)")
    
    pipeline = FeaturePipeline()
    tasks = get_file_list()
    
    if not tasks:
        logging.error("No tasks found. Please check data/mda_texts/")
        return

    logging.info(f"Target Documents: {len(tasks)}")
    
    # --- Phase 1: Embedding Extraction ---
    feature_matrix = []
    valid_meta = []
    
    pbar = tqdm(total=len(tasks), desc="Extracting Embeddings", unit="doc")
    
    for batch in batch_generator(tasks, CONFIG["BATCH_SIZE"]):
        batch_texts = []
        batch_meta = []
        
        # 读取 & 预处理
        for task in batch:
            try:
                with open(task["path"], 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # 跳过空文件或极短文件
                if len(content) < 50:
                    continue
                    
                processed = pipeline.preprocess(content)
                batch_texts.append(processed)
                batch_meta.append(task)
            except Exception as e:
                logging.error(f"Read error {task['path']}: {e}")
        
        # 推理
        if batch_texts:
            try:
                embeddings = pipeline.extract_embeddings(batch_texts)
                feature_matrix.append(embeddings)
                valid_meta.extend(batch_meta)
                pbar.update(len(batch))
            except Exception as e:
                logging.error(f"Inference error: {e}")
                
    pbar.close()
    
    if not feature_matrix:
        logging.error("No valid features extracted.")
        return

    # --- Phase 2: Dimensionality Reduction (PCA) ---
    logging.info("Stacking vectors and initializing PCA...")
    
    X_raw = np.vstack(feature_matrix)
    logging.info(f"   Raw Feature Matrix Shape: {X_raw.shape}")
    
    # PCA 训练
    pca = PCA(n_components=CONFIG["N_COMPONENTS"], random_state=CONFIG["RANDOM_STATE"])
    X_pca = pca.fit_transform(X_raw)
    
    # 检查方差解释率
    explained_ratio = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA Completed. Cumulative Explained Variance: {explained_ratio:.2%}")
    
    if explained_ratio < 0.7:
        logging.warning("Explained variance is low (<70%). Info loss might be high.")

    # --- Phase 3: Export Results ---
    logging.info("Saving structured features...")
    
    results = []
    for idx, meta in enumerate(valid_meta):
        row = {
            "symbol": meta["symbol"],
            "report_year": meta["report_year"]
        }
        # 展开 PCA 特征
        for i in range(CONFIG["N_COMPONENTS"]):
            row[f"pca_{i+1}"] = round(X_pca[idx, i], 6)
        results.append(row)
        
    df = pd.DataFrame(results)
    # 按年份和代码排序
    df.sort_values(by=["report_year", "symbol"], inplace=True)
    
    df.to_csv(CONFIG["OUTPUT_CSV"], index=False)
    
    logging.info(f"Pipeline Done. Output saved to: {CONFIG['OUTPUT_CSV']}")

if __name__ == "__main__":
    main()