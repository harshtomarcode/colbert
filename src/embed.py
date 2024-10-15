import os
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from pgvector.psycopg import register_vector
import psycopg
from pdf import extract_text_from_pdf, chunk_text
import time
import math
import logging
import psutil
from utils import log_memory_usage
import sys
from tqdm import tqdm

# Database connection
db_config = {
    'dbname': os.environ.get('POSTGRES_DB', 'vectordb'),
    'user': os.environ.get('POSTGRES_USER', 'postgres'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'password'),
    'host': os.environ.get('POSTGRES_HOST', 'pgvector-db'),
    'port': os.environ.get('POSTGRES_PORT', '5432')
}

def connect_with_retries(max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            conn = psycopg.connect(**db_config, autocommit=True)
            print("Successfully connected to the database")
            return conn
        except psycopg.OperationalError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Unable to connect to the database.")
                raise

conn = connect_with_retries()

# Set up pgvector
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

# Create table for documents
conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embeddings vector(128)[])')

# Create max_sim function
conn.execute("""
CREATE OR REPLACE FUNCTION max_sim(document vector[], query vector[]) RETURNS double precision AS $$
    WITH queries AS (
        SELECT row_number() OVER () AS query_number, q.query
        FROM (SELECT unnest(query) AS query) AS q
    ),
    documents AS (
        SELECT unnest(document) AS document
        FROM (SELECT document) AS d
    ),
    similarities AS (
        SELECT query_number, 1 - (document <=> query) AS similarity FROM queries CROSS JOIN documents
    ),
    max_similarities AS (
        SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
    )
    SELECT SUM(max_similarity) FROM max_similarities
$$ LANGUAGE SQL
""")

# Set up ColBERT
config = ColBERTConfig(doc_maxlen=220, query_maxlen=32)
checkpoint = Checkpoint('colbert-ir/colbertv2.0', colbert_config=config, verbose=0)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def embed_and_store_pdf(pdf_path, batch_size=128):
    start_time = time.time()
    logging.info(f"Starting to process PDF: {pdf_path}")
    log_memory_usage()

    logging.info("Extracting text from PDF")
    pdf_text = extract_text_from_pdf(pdf_path)
    logging.info(f"Extracted {len(pdf_text)} characters from PDF")
    log_memory_usage()

    logging.info("Creating chunks and processing")
    chunk_generator = chunk_text(pdf_text)
    
    batch = []
    chunk_count = 0
    batch_count = 0
    
    # Create progress bar for embedding and storing
    with tqdm(desc="Embedding and storing", unit="chunk") as pbar:
        for chunk in chunk_generator:
            batch.append(chunk)
            chunk_count += 1
            if len(batch) == batch_size:
                batch_count += 1
                process_batch(batch)
                batch = []
                pbar.update(batch_size)
                
                if batch_count % 10 == 0:  # Log every 10 batches
                    logging.info(f"Processed {chunk_count} chunks ({batch_count} batches)")
                    log_memory_usage()
            
            # Log progress every 1000 chunks
            if chunk_count % 1000 == 0:
                elapsed_time = time.time() - start_time
                chunks_per_second = chunk_count / elapsed_time
                logging.info(f"Processed {chunk_count} chunks. "
                             f"Rate: {chunks_per_second:.2f} chunks/second.")
                pbar.set_postfix({'Memory': f"{psutil.virtual_memory().percent}%"})
        
        if batch:
            batch_count += 1
            process_batch(batch)
            pbar.update(len(batch))
    
    total_time = time.time() - start_time
    logging.info(f"Completed embedding and storing {chunk_count} chunks from {pdf_path}")
    logging.info(f"Total processing time: {total_time/60:.2f} minutes")
    log_memory_usage()

def process_batch(batch):
    logging.info(f"Generating embeddings for batch of {len(batch)} chunks")
    
    # Generate embeddings for the batch
    logging.info("Generating embeddings for batch")
    doc_embeddings = checkpoint.docFromText(batch, keep_dims=False)
    logging.info("Embeddings generated")
    log_memory_usage()

    # Store embeddings in the database
    logging.info("Storing embeddings in database")
    for content, embeddings in zip(batch, doc_embeddings):
        embeddings = [e.numpy() for e in embeddings]
        conn.execute('INSERT INTO documents (content, embeddings) VALUES (%s, %s)', (content, embeddings))
    logging.info("Batch stored in database")
    log_memory_usage()

def search_similar_documents(query, limit=5):
    query_embeddings = [e.numpy() for e in checkpoint.queryFromText([query])[0]]
    result = conn.execute('SELECT content, max_sim(embeddings, %s) AS max_sim FROM documents ORDER BY max_sim DESC LIMIT %s', (query_embeddings, limit)).fetchall()
    return result
