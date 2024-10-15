from pdf import extract_text_from_pdf
from embed import embed_and_store_pdf, search_similar_documents
from llm import generate_response


def process_pdf(pdf_path):
    embed_and_store_pdf(pdf_path)

def get_response(query):
    similar_docs = search_similar_documents(query)
    context = "\n".join([doc[0] for doc in similar_docs])
    response, _ = generate_response("src/prompts/response.yml", context, [{"role": "user", "content": query}])
    return response
