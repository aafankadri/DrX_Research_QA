import re
import csv

log_file = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\performance.log"
csv_file = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\performance_data.csv"

# Regex to capture:
# Timestamp | Task | Tokens | Elapsed Time | Tokens/sec
pattern = re.compile(
    r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - .*?(Embedding|Translation|Summarization|RAG Q&A).*?\| (?P<tokens>[\d,]+) tokens.*?\| (?P<elapsed>[\d.]+)s elapsed \| (?P<tps>[\d.]+) tokens/sec"
)

# Storage list
rows = []

# Read and match
with open(log_file, "r", encoding="utf-8") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            data = match.groupdict()
            rows.append({
                "Datetime": data["datetime"],
                "Task": match.group(2),
                "Tokens": int(data["tokens"].replace(",", "")),
                "Time (s)": float(data["elapsed"]),
                "Tokens/sec": float(data["tps"])
            })

# Write to CSV
with open(csv_file, "w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=["Datetime", "Task", "Tokens", "Time (s)", "Tokens/sec"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Export complete. CSV saved as {csv_file}")
