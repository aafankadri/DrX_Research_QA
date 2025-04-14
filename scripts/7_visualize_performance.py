import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

log_file = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\performance.log"

output_dir = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\charts"
os.makedirs(output_dir, exist_ok=True)

# Regex to extract:
# Task | tokens | elapsed time | tokens/sec
pattern = re.compile(r"(Embedding|Translation|Summarization|RAG Q&A).*\|\s([\d,]+)\s+tokens.*\|\s([\d.]+)s elapsed\s\|\s([\d.]+) tokens/sec")

# Store: {task: [tokens/sec]}
performance_data = defaultdict(list)

# Parse the log
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            task, tokens, elapsed, tps = match.groups()
            performance_data[task].append(float(tps))

# Plot separate graphs
for task, tps_values in performance_data.items():
    plt.figure(figsize=(8, 5))
    plt.plot(tps_values, marker='o', linestyle='-', color='blue')
    plt.title(f"📈 Tokens/sec for {task}")
    plt.xlabel("Run #")
    plt.ylabel("Tokens/sec")
    plt.grid(True)
    plt.tight_layout()
    
    # Save with lowercase filename
    safe_name = task.lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/{safe_name}_performance.png")
    plt.close()

print("✅ Saved individual charts in 'charts/' folder.")