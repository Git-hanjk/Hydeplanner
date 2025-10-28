# demo/pdf_processing_module.py
import hashlib
import json
import os
from io import BytesIO
from typing import Tuple

import faiss
import pdfplumber
import requests
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "pdf_embeddings_cache")


def _get_cache_paths(pdf_url: str) -> Tuple[str, str]:
    """Return filesystem paths for the FAISS index and metadata files."""
    hashed = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()
    base_path = os.path.join(CACHE_DIR, hashed)
    return f"{base_path}.faiss", f"{base_path}.json"

# --- Model Loading (Cached) ---

@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Failed to load the embedding model: {e}")
        return None

# --- Core PDF Processing Logic ---

def process_pdf_with_embeddings(pdf_url: str, query: str, num_relevant_chunks: int = 5, save_embeddings: bool = False) -> str:
    """
    Processes a PDF from a URL using embeddings to find the most relevant text chunks.

    Args:
        pdf_url (str): The URL of the PDF file.
        query (str): The user query to find relevant information for.
        num_relevant_chunks (int): The number of top relevant chunks to return.
        save_embeddings (bool): Whether to cache the FAISS index and chunks for reuse.

    Returns:
        str: A concatenated string of the most relevant text chunks, or an error message.
    """
    model = load_embedding_model()
    if model is None:
        return "Error: Embedding model is not available."

    index = None
    chunks = []
    index_path = metadata_path = None

    if save_embeddings:
        index_path, metadata_path = _get_cache_paths(pdf_url)
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                cached_index = faiss.read_index(index_path)
                with open(metadata_path, "r") as meta_file:
                    metadata = json.load(meta_file)
                if metadata.get("model_name") == MODEL_NAME:
                    chunks = metadata.get("chunks", [])
                    if len(chunks) != cached_index.ntotal:
                        print(f"Cached chunk count mismatch for {pdf_url}; rebuilding embeddings.")
                    else:
                        index = cached_index
                else:
                    print(f"Embedding model changed since cache creation for {pdf_url}; rebuilding embeddings.")
            except Exception as cache_error:
                print(f"Failed to load cached embeddings for {pdf_url}: {cache_error}")
                index = None
                chunks = []

    try:
        if index is None or not chunks:
            response = requests.get(pdf_url, timeout=20)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)

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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
            chunks = text_splitter.split_text(full_text)

            if not chunks:
                return "Warning: Text was extracted but could not be split into chunks."

            with st.spinner(f"Analyzing {len(chunks)} text sections from the PDF..."):
                chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
                chunk_embeddings_np = chunk_embeddings.cpu().detach().numpy()
                index = faiss.IndexFlatL2(chunk_embeddings_np.shape[1])
                index.add(chunk_embeddings_np)

            if save_embeddings and index is not None and index_path and metadata_path:
                try:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    faiss.write_index(index, index_path)
                    with open(metadata_path, "w") as meta_file:
                        json.dump({"model_name": MODEL_NAME, "chunks": chunks}, meta_file)
                except Exception as cache_save_error:
                    st.warning(f"Failed to cache PDF embeddings: {cache_save_error}")

        if index is None or not chunks:
            return "Warning: Failed to build embeddings for the PDF."

        query_embedding = model.encode(query, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().detach().numpy().reshape(1, -1)

        k = min(num_relevant_chunks, len(chunks))
        if k < 1:
            k = len(chunks)

        distances, indices = index.search(query_embedding_np, k)
        relevant_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

        if not relevant_chunks:
            return "Warning: No relevant sections were found in the PDF."

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
