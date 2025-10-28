
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

class LocalSearch:
    def __init__(self, index_path=None):
        if index_path is None:
            # Get the absolute path of the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.index_path = os.path.join(script_dir, "faiss_index")
        else:
            self.index_path = index_path
            
        self.embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Please run ingest.py first.")
        
        print("Loading FAISS index...")
        return FAISS.load_local(self.index_path, self.embedding_function, allow_dangerous_deserialization=True)

    def search(self, query, k=5):
        """
        Searches the FAISS index for the most similar documents to the query.
        """
        print(f"Searching for '{query}' in local documents...")
        results = self.vector_store.similarity_search(query, k=k)
        
        # Format the results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return formatted_results

if __name__ == '__main__':
    # Example usage
    try:
        local_search = LocalSearch()
        query = "What is the main idea of the document?"
        search_results = local_search.search(query)
        
        if search_results:
            print(f"\nFound {len(search_results)} results for '{query}':")
            for i, result in enumerate(search_results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Content: {result['content']}")
                if result['metadata']:
                    print(f"Source: {result['metadata'].get('source', 'N/A')}")
        else:
            print("No results found.")
            
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
