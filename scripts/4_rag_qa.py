import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --- Configs ---
VECTOR_DB_DIR = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\vectorstore"
TOP_K = 5
LLM_MODEL_PATH = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\models\llama-2-7b.Q4_K_M.gguf"  # adjust based on your model

# --- Load vector index and metadata ---
index = faiss.read_index(os.path.join(VECTOR_DB_DIR, "index.faiss"))
with open(os.path.join(VECTOR_DB_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --- Load Embedding Model ---
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# --- Load LLaMA LLM locally ---
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=4096,
    n_threads=6,
    temperature=0.2,
    top_p=0.95,
    verbose=False
)

def get_relevant_chunks(query, top_k=TOP_K):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    retrieved = [metadata[i] for i in indices[0]]
    return retrieved

def build_prompt(query, context_chunks):
    context_text = "\n---\n".join([f"{c['text']}" for c in context_chunks])
    prompt = f"""You are an expert assistant helping to analyze research documents.

Use the following context to answer the user's question. If the answer is not in the context, say you don’t know.

Context:
{context_text}

Question:
{query}

Answer:"""
    return prompt

def rag_qa(query):
    context_chunks = get_relevant_chunks(query)
    prompt = build_prompt(query, context_chunks)
    output = llm(prompt, stop=["\n\n", "User:"])
    return output['choices'][0]['text'].strip(), context_chunks

# --- Example Usage ---
if __name__ == "__main__":
    while True:
        user_q = input("\nAsk a question (or 'exit'): ")
        if user_q.lower() == "exit":
            break
        answer, used_chunks = rag_qa(user_q)
        print("\n📘 Answer:", answer)
        print("\n📚 Context used:")
        for chunk in used_chunks:
            print(f"\n→ From {chunk['source_file']} (Page {chunk['page']}):\n{chunk['text'][:200]}...")
