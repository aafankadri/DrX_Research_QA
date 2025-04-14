import os
import fitz  # PyMuPDF
import docx
import pandas as pd

SUPPORTED_EXT = [".pdf", ".docx", ".csv", ".xlsx", ".xls", ".xlsm"]

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    content = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        content.append(f"[Page {page_num}]\n{text}")
    return "\n".join(content)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def dataframe_to_markdown(df):
    from tabulate import tabulate
    return tabulate(df, headers="keys", tablefmt="github", showindex=False)

def extract_text_from_excel(file_path):
    text = []
    excel_data = pd.read_excel(file_path, sheet_name=None)
    for sheet, df in excel_data.items():
        text.append(f"[Sheet: {sheet}]")
        text.append(dataframe_to_markdown(df))
    return "\n\n".join(text)

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return dataframe_to_markdown(df)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".xlsx", ".xls", ".xlsm"]:
        return extract_text_from_excel(file_path)
    elif ext == ".csv":
        return extract_text_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Example usage
if __name__ == "__main__":
    input_dir = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\data"
    output_dir = r"C:\MarkyticsProjectCode\osos\DrX_Research_QA\extracted"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.splitext(filename)[1].lower() in SUPPORTED_EXT:
            print(f"Extracting from {filename}")
            try:
                extracted = extract_text(filepath)
                with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                    f.write(extracted)
            except Exception as e:
                print(f"Failed to extract from {filename}: {e}")
