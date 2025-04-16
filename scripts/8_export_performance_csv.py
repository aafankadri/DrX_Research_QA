import re
import csv
import os

# --- Base Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log_file = os.path.join(BASE_DIR, "performance.log")
csv_file = os.path.join(BASE_DIR, "performance_data.csv")

# Regex to capture:
# Timestamp | Task | Tokens | Elapsed Time | Tokens/sec
task_pattern = re.compile(
    r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - .*?(Embedding|Translation|Summarization|RAG Q&A).*?\| (?P<tokens>[\d,]+) tokens.*?\| (?P<elapsed>[\d.]+)s elapsed \| (?P<tps>[\d.]+) tokens/sec"
)

# Regex to capture ROUGE scores
rouge_pattern = re.compile(
    r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - .*?ROUGE \| Avg ROUGE-1: (?P<rouge1>[\d.]+) \| Avg ROUGE-L: (?P<rougeL>[\d.]+)"
)

# Storage list
rows = []

# Temporary storage for ROUGE scores
rouge_scores = {}

# Read and match
with open(log_file, "r", encoding="utf-8") as file:
    for line in file:
        # Match task logs
        task_match = task_pattern.search(line)
        if task_match:
            data = task_match.groupdict()
            datetime = data["datetime"]
            rows.append({
                "Datetime": datetime,
                "Task": task_match.group(2),
                "Tokens": int(data["tokens"].replace(",", "")),
                "Time (s)": float(data["elapsed"]),
                "Tokens/sec": float(data["tps"]),
                "ROUGE-1": rouge_scores.get(datetime, {}).get("rouge1"),
                "ROUGE-L": rouge_scores.get(datetime, {}).get("rougeL"),
            })
        
        # Match ROUGE logs
        rouge_match = rouge_pattern.search(line)
        if rouge_match:
            rouge_data = rouge_match.groupdict()
            rouge_scores[rouge_data["datetime"]] = {
                "rouge1": float(rouge_data["rouge1"]),
                "rougeL": float(rouge_data["rougeL"]),
            }

# Write to CSV
with open(csv_file, "w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=["Datetime", "Task", "Tokens", "Time (s)", "Tokens/sec", "ROUGE-1", "ROUGE-L"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Export complete. CSV saved as {csv_file}")
