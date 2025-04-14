# 🧠 Dr. X Research Q&A System — OSOS AI Technical Test

This repository contains my solution to the OSOS AI NLP Technical Challenge. The task was to analyze Dr. X’s multi-format publications and build a complete RAG-based Q&A system using only **local models** and **offline vector databases**.

---

## 📦 Project Structure
DrX_Research_QA/ 
│── data/   # Store Dr. X's publications 
├── chunks/ # Chunked text with metadata 
├── extracted/ # Raw extracted text files 
│
├── models/                   # For local LLMs or embedding models
├── vectorstore/ # FAISS index + metadata 
│
├── translated/ # Translated chunks 
├── summaries/ # Summarized chunks 
│ 
├── scripts/
│   ├── 1_extract_text.py     # File extraction from PDFs, DOCX, XLSX, CSV
│   ├── 2_chunk_text.py       # Tokenize and chunk using cl100k_base
│   ├── 3_embed_and_store.py  # Embedding + vector DB creation
│   ├── 4_rag_qa.py           # RAG Q&A system with local LLaMA
│   ├── 5_translate.py        # Translation tool
│   ├── 6_summarize.py        # Summarization + ROUGE
│   └── utils.py              # Utility functions
│
├── performance.log 
├── README.md 
└── requirements.txt



---

## ✅ Features

- 📄 Multi-format file processing: PDF, DOCX, CSV, XLSX, XLSM
- 🧩 Token-based chunking with `cl100k_base` tokenizer
- 🧠 Vector embeddings using `nomic-ai/nomic-embed-text-v1`
- 🧠 Local RAG Q&A using `llama-cpp-python` with LLaMA GGUF model
- 🌍 Multilingual translation to English/Arabic using `facebook/nllb-200-distilled-600M`
- ✂️ Summarization using `facebook/bart-large-cnn`
- 📊 ROUGE-1 and ROUGE-L evaluation
- 🚀 Tokens/sec logging for all major steps

---

## 🧪 Performance Logging

All performance is logged to `performance.log`:
- Total tokens processed
- Time taken
- Tokens per second
- Per-file and per-query level

---

## 🔍 Local Models Used

| Task            | Model                                    |
|-----------------|-------------------------------------------|
| Embedding       | `nomic-ai/nomic-embed-text-v1`           |
| Translation     | `facebook/nllb-200-distilled-600M`       |
| Summarization   | `facebook/bart-large-cnn`                |
| RAG LLM         | `llama-2-7b.Q4_K_M.gguf` (via llama.cpp) |

---

## 🛠️ How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt

2. Put your .pdf, .docx, .xlsx, etc. into the data/ folder

3. Run each step:
    python 1_extract_text.py
    python 2_chunk_text.py
    python 3_embed_and_store.py
    python 4_rag_qa.py
    python 5_translate.py
    python 6_summarize.py

## 📜 Requirements

faiss-cpu
sentence-transformers
transformers
langdetect
tiktoken
llama-cpp-python
rouge-score
evaluate
pandas
python-docx
openpyxl
tqdm


## 🧠 Notes

*   All models run locally with no external APIs or internet required.
*   No vision-based tools were used for PDFs — purely NLP-based extraction.
*   Vector DB is built using FAISS for offline efficiency.
