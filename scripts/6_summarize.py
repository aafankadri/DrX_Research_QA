import os
import json
import time
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm import tqdm
import tiktoken

# --- Base Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(BASE_DIR, "performance.log")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")  # or 'translated'
SUMMARY_DIR = os.path.join(BASE_DIR, "summaries")
os.makedirs(SUMMARY_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# Load summarization model (e.g., BART or T5)
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_text(text, max_tokens=512):
    return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]

def summarize_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    start = time.time()

    summaries = []
    for chunk in chunks:
        text = chunk["text"]
        if len(text.strip().split()) < 50:
            continue  # skip very small chunks

        try:
            summary = summarize_text(text)
            chunk["summary"] = summary
            summaries.append(chunk)
            total_tokens += len(tokenizer.encode(text))
        except Exception as e:
            print(f"Failed to summarize chunk: {e}")
            continue
    
    elapsed = time.time() - start
    tps = total_tokens / elapsed if elapsed > 0 else 0

    log_msg = (
        f"✂️ Summarization | File: {os.path.basename(json_path)} | "
        f"{len(summaries)} summaries | {total_tokens} tokens | "
        f"{elapsed:.2f}s elapsed | {tps:.2f} tokens/sec"
    )
    logging.info(log_msg)
    print(log_msg)

    return summaries

def evaluate_rouge(chunks_with_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []

    for chunk in chunks_with_summaries:
        if "summary" in chunk:
            reference = chunk["text"]
            generated = chunk["summary"]
            score = scorer.score(reference, generated)
            scores.append(score)

    if not scores:  # Handle empty scores
        return {"rouge1": 0.0, "rougeL": 0.0}

    # Average scores
    avg_scores = {
        "rouge1": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
        "rougeL": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
    }

    # Log average ROUGE
    logging.info(
        f"🔍 ROUGE | Avg ROUGE-1: {avg_scores['rouge1']:.4f} | Avg ROUGE-L: {avg_scores['rougeL']:.4f}"
    )
    
    return avg_scores

# --- Main Flow ---
if __name__ == "__main__":
    for file in tqdm(os.listdir(CHUNKS_DIR)):
        if not file.endswith(".json"):
            continue

        in_path = os.path.join(CHUNKS_DIR, file)
        print(f"✂️ Summarizing {file}")
        summarized_chunks = summarize_chunks(in_path)

        rouge = evaluate_rouge(summarized_chunks)
        print(f"📊 ROUGE scores for {file}: ROUGE-1: {rouge['rouge1']:.3f}, ROUGE-L: {rouge['rougeL']:.3f}")

        out_path = os.path.join(SUMMARY_DIR, file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summarized_chunks, f, indent=2, ensure_ascii=False)
