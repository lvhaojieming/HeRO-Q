import json
import os
import re
import fasttext
import textstat
import nltk
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter
import unicodedata
import html


INPUT_DIR = "/home/zjh/dataset/C4_Qwen/en"
INPUT_FILE = os.path.join(INPUT_DIR, "c4_deepspeed_merged.jsonl") 
OUTPUT_FILE = os.path.join(INPUT_DIR, "c4_cleaned_full.jsonl")
FASTTEXT_MODEL_PATH = "/home/zjh/model/lid.176.bin"
MIN_LENGTH = 100            
MAX_LENGTH = 100000         
MIN_AVG_LINE_LENGTH = 10    
MAX_SYMBOL_RATIO = 0.15     # 稍微放宽
MIN_READABILITY_SCORE = 0   # 暂时设为0进行排查
NUM_PERM = 128              
LSH_THRESHOLD = 0.8         

ft_model = None
def init_worker():
    global ft_model
    try:
        ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    except Exception as e:
        pass

def clean_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    url_pattern = r'(https?://\S+|www\.\S+|\b\w+\.(?:com|org|net|edu|gov|io|me|info)\b/\S*)'
    text = re.sub(url_pattern, '', text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s.,!?;:\'"()\[\]%$&-]','', text)
    text = re.sub(r'([!?,;])\1+', r'\1', text)
    text = re.sub(r'-{3,}', '--', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_minhash(text):
    tokens = text.lower().split() 
    m = MinHash(num_perm=NUM_PERM)
    for t in tokens:
        m.update(t.encode('utf8'))
    return m

def calculate_quality_metrics(text):
    if not (MIN_LENGTH <= len(text) <= MAX_LENGTH):
        return None, f"filter_length_{len(text)}"
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) == 0: return None, "filter_empty_length"
    ratio = symbols / len(text)
    if ratio > MAX_SYMBOL_RATIO:
        return None, f"filter_symbol_ratio_{ratio:.2f}"
    if ft_model:
        try:
            pred = ft_model.predict(text.replace("\n", " "))
            lang = pred[0][0].replace("__label__", "")
            score = pred[1][0]
            if lang not in ["en", "eng"] or score < 0.3:
                return None, f"filter_lang_{lang}_{score:.2f}"
        except Exception as e:
            return None, f"filter_lang_error_{str(e)}"
    
    try:
        readability = textstat.flesch_reading_ease(text)
        if readability < MIN_READABILITY_SCORE:
            return None, f"filter_readability_{readability:.2f}"
    except Exception as e:
        pass 
        
    return {}, "pass"

def process_line(line):
    try:
        item = json.loads(line)
        text = item.get("text", "")
        
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return {"status": "filter_empty"}
        
        metrics, reason = calculate_quality_metrics(cleaned_text)
        if reason != "pass":
            return {"status": reason}
            
        item['text'] = cleaned_text
        mh = get_minhash(cleaned_text)
        
        return {
            "status": "pass",
            "item": item,
            "minhash": mh
        }
    except Exception as e:
        return {"status": f"error_{str(e)}"}

def main():
    print("Starting comprehensive data cleaning...")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # 读取所有行
    lines = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    total_count = len(lines)
    print(f"Loaded {total_count} lines. Processing...")

    reasons = Counter()

    with Pool(processes=cpu_count(), initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_line, lines), total=total_count, desc="Cleaning & Filtering"))
    
    print("\nFiltering Stats:")
    for res in results:
        if res:
            reasons[res['status']] += 1
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    print("\nPerforming Deduplication (LSH) and Writing...")
    
    seen_hashes = set()
    kept_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for res in tqdm(results, desc="Deduplicating"):
            if not res or res['status'] != "pass":
                continue
                
            item = res['item']
            mh = res['minhash']
            
            # 使用 MinHash 摘要进行近似去重
            digest = str(mh.digest())
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            
            # 移除 URL 字段 (如果存在)
            item.pop('url', None)
            
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept_count += 1
            
    print(f"Done. Kept {kept_count}/{total_count} documents.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
