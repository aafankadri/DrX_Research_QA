import os
import json
import nltk
import tiktoken

nltk.download('punkt')

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHUNK_TOKEN_LIMIT = 500
CHUNK_OVERLAP_SENTENCES = 1


tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize_text(text):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

def smart_chunk_text(text):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        token_len = len(sentence_tokens)

        if current_tokens + token_len <= CHUNK_TOKEN_LIMIT:
            current_chunk.append(sentence)
            current_tokens += token_len
        else:
            chunks.append(" ".join(current_chunk))
            # Overlap: take last few sentences from current chunk
            current_chunk = current_chunk[-CHUNK_OVERLAP_SENTENCES:] + [sentence]
            current_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_tokens(tokens, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        start += chunk_size - overlap
    return chunks

def process_extracted_text(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    pages = text.split("[Page ")
    chunks_metadata = []

    for page_block in pages:
        if not page_block.strip():
            continue

        try:
            page_num = int(page_block.split("]")[0])
            page_text = page_block.split("]", 1)[1]
        except:
            page_num = -1
            page_text = page_block

        smart_chunks = smart_chunk_text(page_text)
        for i, chunk in enumerate(smart_chunks):
            chunks_metadata.append({
                "source_file": filename,
                "page": page_num,
                "chunk_number": i + 1,
                "text": chunk
            })

    return chunks_metadata

# def process_extracted_text(file_path):
#     filename = os.path.basename(file_path)
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()

#     # Optional: Try to track pages
#     pages = text.split("[Page ")
#     chunks_metadata = []

#     for page_block in pages:
#         if not page_block.strip():
#             continue

#         if page_block.startswith("1]"):
#             page_num = 1
#             page_text = page_block[2:]
#         else:
#             try:
#                 page_num = int(page_block.split("]")[0])
#                 page_text = page_block.split("]", 1)[1]
#             except:
#                 page_num = -1
#                 page_text = page_block

#         tokens = tokenize_text(page_text)
#         token_chunks = chunk_tokens(tokens)

#         for i, chunk_tokens_ in enumerate(token_chunks):
#             chunk_text = detokenize(chunk_tokens_)
#             chunks_metadata.append({
#                 "source_file": filename,
#                 "page": page_num,
#                 "chunk_number": i + 1,
#                 "text": chunk_text
#             })

#     return chunks_metadata

if __name__ == "__main__":
    input_dir = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\extracted"
    output_dir = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\chunks"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not filename.endswith(".txt"):
            continue

        print(f"Chunking {filename}")
        chunks = process_extracted_text(filepath)

        out_file = os.path.join(output_dir, f"{filename}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
