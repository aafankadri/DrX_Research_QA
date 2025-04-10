import os
import json
import torch
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Config ---
CHUNKS_DIR = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\chunks"
TRANSLATED_DIR = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\translated"
os.makedirs(TRANSLATED_DIR, exist_ok=True)

# Load NLLB model
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# NLLB language codes
lang_map = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    # fallback for common languages (expandable)
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "hi": "hin_Deva",
    "de": "deu_Latn"
}

def detect_lang(text):
    try:
        detected = detect(text)
        return detected
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "unknown"

def translate_text(text, src_lang_code, tgt_lang_code):
    if src_lang_code == tgt_lang_code:
        return text  # no need to translate
    tokenized = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(
        **tokenized,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
        max_length=512
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_chunks(json_file, target="en"):
    target_code = lang_map[target]
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    translated_chunks = []

    for chunk in chunks:
        original_text = chunk["text"]
        detected_lang = detect_lang(original_text)
        source_code = lang_map.get(detected_lang, "eng_Latn")

        try:
            translated = translate_text(original_text, source_code, target_code)
        except Exception as e:
            print(f"Failed to translate chunk: {e}")
            translated = original_text

        chunk["translated_text"] = translated
        chunk["source_lang"] = detected_lang
        translated_chunks.append(chunk)

    return translated_chunks

# --- Example Usage ---
if __name__ == "__main__":
    for file in os.listdir(CHUNKS_DIR):
        if not file.endswith(".json"):
            continue
        filepath = os.path.join(CHUNKS_DIR, file)
        print(f"🔄 Translating {file} → English")
        output = translate_chunks(filepath, target="en")

        out_path = os.path.join(TRANSLATED_DIR, f"{file}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
