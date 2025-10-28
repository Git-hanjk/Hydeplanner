

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. Load Documents
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            documents.extend(loader.load())
    return documents

# 2. Split Documents
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

# 3. Create Embeddings and Store in FAISS
def create_vector_store(splits, index_path):
    # Use a local sentence transformer model
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store in memory
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embedding_function
    )
    
    # Save the vector store to a local file
    vectorstore.save_local(index_path)
    print(f"Vector store created and saved in '{index_path}'")
    return vectorstore

def run_ingestion():
    """Loads documents, splits them, and creates a FAISS vector store."""
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration using absolute paths
    docs_folder = os.path.join(script_dir, "local_docs")
    index_path = os.path.join(script_dir, "faiss_index")

    # Check if the docs folder exists
    if not os.path.exists(docs_folder):
        print(f"Error: The directory '{docs_folder}' does not exist. Please create it and add your documents.")
        return

    # Load the documents
    print(f"Loading documents from '{docs_folder}'...")
    documents = load_documents(docs_folder)
    if not documents:
        print("No documents found to process.")
        return
    
    print(f"Loaded {len(documents)} documents.")

    # Split the documents
    print("Splitting documents into chunks...")
    splits = split_documents(documents)
    print(f"Created {len(splits)} document chunks.")

    # Create and save the vector store
    print("Creating FAISS vector store...")
    create_vector_store(splits, index_path)

def main():
    run_ingestion()

if __name__ == "__main__":
    main()
