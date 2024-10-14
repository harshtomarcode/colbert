from pdf import extract_text_from_pdf
from embed import embed_and_store_pdf, search_similar_documents
from llm import generate_response

user_msg = "Hello"
print(generate_response([{"role": "user", "content": user_msg}]))

pdf_path = "path/to/your/pdf/file.pdf"
process_pdf(pdf_path)

query = "Your search query here"
results = search_similar_documents(query)
for row in results:
    print(row)

def process_pdf(pdf_path):
    embed_and_store_pdf(pdf_path)

def get_response(query):
    similar_docs = search_similar_documents(query)
    context = "\n".join([doc[0] for doc in similar_docs])
    response = generate_response("You are a helpful assistant.", context, query)
    return response
