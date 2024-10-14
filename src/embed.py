import os
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from pgvector.psycopg import register_vector
import psycopg
from pdf import extract_text_from_pdf, chunk_text

# Database connection
db_config = {
    'dbname': os.environ.get('POSTGRES_DB', 'vectordb'),
    'user': os.environ.get('POSTGRES_USER', 'postgres'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'password'),
    'host': os.environ.get('POSTGRES_HOST', 'pgvector-db'),
    'port': os.environ.get('POSTGRES_PORT', '5432')
}

conn = psycopg.connect(**db_config, autocommit=True)

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
        SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query)
    ),
    documents AS (
        SELECT unnest(document) AS document
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

def embed_and_store_pdf(pdf_path):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    chunks = chunk_text(pdf_text)
    
    # Generate embeddings
    doc_embeddings = checkpoint.docFromText(chunks, keep_dims=False)
    
    # Store embeddings in the database
    for content, embeddings in zip(chunks, doc_embeddings):
        embeddings = [e.numpy() for e in embeddings]
        conn.execute('INSERT INTO documents (content, embeddings) VALUES (%s, %s)', (content, embeddings))
    
    print(f"Embedded and stored {len(chunks)} chunks from {pdf_path}")

def search_similar_documents(query, limit=5):
    query_embeddings = [e.numpy() for e in checkpoint.queryFromText([query])[0]]
    result = conn.execute('SELECT content, max_sim(embeddings, %s) AS max_sim FROM documents ORDER BY max_sim DESC LIMIT %s', (query_embeddings, limit)).fetchall()
    return result
