import pymupdf


def load_pdf(pdf_path):
    try:
        return pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    doc = load_pdf(pdf_path)
    if doc is None:
        return ""
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks



