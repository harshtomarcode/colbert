import pymupdf
import logging
from utils import log_memory_usage
from tqdm import tqdm


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
    text_length = len(text)
    total_chunks = (text_length - overlap) // (chunk_size - overlap) + 1
    
    logging.info(f"Chunking text of length {text_length} into approximately {total_chunks} chunks")
    
    start = 0
    chunk_count = 0
    with tqdm(total=total_chunks, desc="Chunking text", unit="chunk") as pbar:
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            yield text[start:end]
            start = end - overlap
            chunk_count += 1
            if chunk_count % 100 == 0:  # Update progress bar every 100 chunks
                pbar.update(100)
    
    # Update progress bar for any remaining chunks
    pbar.update(total_chunks - ((chunk_count // 100) * 100))
    logging.info(f"Finished chunking text into {chunk_count} chunks")


def get_pdf_page_count(pdf_path):
    doc = load_pdf(pdf_path)
    if doc is None:
        return 0
    return len(doc)


def get_pdf_page_text(pdf_path, page_number):
    doc = load_pdf(pdf_path)
    if doc is None or page_number < 0 or page_number >= len(doc):
        return ""
    return doc[page_number].get_text()




