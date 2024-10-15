import streamlit as st
import tempfile
import os
from app import process_pdf, get_response
from pdf import load_pdf, get_pdf_page_count, get_pdf_page_text

st.set_page_config(layout="wide")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.title("PDF Chat Application")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chat")
        display_chat()

        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            display_chat()

            with st.spinner("Generating response..."):
                response = get_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
            display_chat()

    with col2:
        st.subheader("PDF Viewer")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            st.session_state.pdf_path = tmp_file_path
            with st.spinner("Processing PDF..."):
                process_pdf(tmp_file_path)

            num_pages = get_pdf_page_count(tmp_file_path)

            page_number = st.number_input("Page number", min_value=1, max_value=num_pages, value=1)
            page_text = get_pdf_page_text(tmp_file_path, page_number - 1)
            st.text_area("PDF Content", value=page_text, height=400)

if __name__ == "__main__":
    main()
