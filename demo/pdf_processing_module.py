# demo/pdf_processing_module.py
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pdfplumber
from io import BytesIO
import requests

# --- Model Loading (Cached) ---

@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load the embedding model: {e}")
        return None

# --- Core PDF Processing Logic ---

def process_pdf_with_embeddings(pdf_url: str, query: str, num_relevant_chunks: int = 5) -> str:
    """
    Processes a PDF from a URL using embeddings to find the most relevant text chunks.

    Args:
        pdf_url (str): The URL of the PDF file.
        query (str): The user query to find relevant information for.
        num_relevant_chunks (int): The number of top relevant chunks to return.

    Returns:
        str: A concatenated string of the most relevant text chunks, or an error message.
    """
    model = load_embedding_model()
    if model is None:
        return "Error: Embedding model is not available."

    try:
        # 1. Download PDF content
        response = requests.get(pdf_url, timeout=20)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)

        # 2. Extract text from PDF
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            if not pdf.pages:
                return "Warning: PDF is empty or could not be read."
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        
        if not full_text.strip():
            return "Warning: No text could be extracted from the PDF."

        # 3. Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text)
        
        if not chunks:
            return "Warning: Text was extracted but could not be split into chunks."

        # 4. Create embeddings for the chunks
        with st.spinner(f"Analyzing {len(chunks)} text sections from the PDF..."):
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            
            # 5. Build a FAISS index for fast searching
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
            index.add(chunk_embeddings.cpu().detach().numpy())

            # 6. Create embedding for the query and search the index
            query_embedding = model.encode(query, convert_to_tensor=True)
            query_embedding_np = query_embedding.cpu().detach().numpy().reshape(1, -1)

            # Search for the k most similar chunks
            distances, indices = index.search(query_embedding_np, k=min(num_relevant_chunks, len(chunks)))

            # 7. Retrieve and concatenate the relevant chunks
            relevant_chunks = [chunks[i] for i in indices[0]]
            
            # Add a separator for clarity
            separator = "\n\n---\n[Relevant Section]\n---\n\n"
            return separator.join(relevant_chunks)

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to download PDF from URL '{pdf_url}'. Reason: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing the PDF with embeddings: {e}"

if __name__ == '__main__':
    # Example usage for testing the module directly
    st.title("PDF Embedding Search Test")

    # A public PDF URL for testing
    test_pdf_url = st.text_input(
        "Enter a PDF URL:", 
        "https://arxiv.org/pdf/1706.03762.pdf" # "Attention Is All You Need" paper
    )
    test_query = st.text_area(
        "Enter your query about the PDF:", 
        "What is the architecture of the Transformer model?"
    )

    if st.button("Process PDF"):
        if not test_pdf_url or not test_query:
            st.warning("Please provide both a PDF URL and a query.")
        else:
            with st.spinner("Processing PDF with embedding-based search..."):
                result = process_pdf_with_embeddings(test_pdf_url, test_query)
                st.subheader("Most Relevant Information Found:")
                st.markdown(result)
