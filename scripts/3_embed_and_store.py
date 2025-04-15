import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time
import tiktoken
import logging

# Setup logging
log_path = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\performance.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # append to file
)   

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
CHUNKS_DIR = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\chunks"
VECTOR_DB_DIR = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\vectorstore"
CACHE_DIR = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\cache"
EMBEDDING_DIM = 768  # Depends on the model used

# Load Nomic model locally (must be available in your HF cache or downloaded)
model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, cache_folder=CACHE_DIR)

# Store metadata separately
metadata_list = []
vectors = []

def embed_chunks_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    start = time.time()
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    elapsed = time.time() - start
    
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    log_msg = (
        f"🧠 Embedding | File: {os.path.basename(json_path)} | "
        f"{len(texts)} chunks | {total_tokens} tokens | "
        f"{elapsed:.2f}s elapsed | {tokens_per_sec:.2f} tokens/sec"
    )
    logging.info(log_msg)
    print(log_msg)

    return embeddings, chunks

if __name__ == "__main__":
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print("Embedding text chunks and building FAISS index...")
    for filename in tqdm(os.listdir(CHUNKS_DIR)):
        if not filename.endswith(".json"):
            continue
        json_path = os.path.join(CHUNKS_DIR, filename)
        embeddings, chunks = embed_chunks_from_json(json_path)

        vectors.extend(embeddings)
        metadata_list.extend(chunks)

    vectors_np = np.array(vectors).astype("float32")
    
    # Build FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors_np)

    # Save FAISS index
    faiss.write_index(index, os.path.join(VECTOR_DB_DIR, "index.faiss"))

    # Save metadata
    with open(os.path.join(VECTOR_DB_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"Stored {len(vectors_np)} embeddings in vector DB.")
