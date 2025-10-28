# hyde_search_module.py
import traceback
import requests
import re
import string
import concurrent.futures
import time
import threading
import ssl
import logging
from io import BytesIO
from typing import Tuple, List, Dict

import streamlit as st
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import pdfplumber
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from pdf_processing_module import process_pdf_with_embeddings
from settings import Environment

# --- PDFplumber Warning Silencing ---
# Suppress noisy warnings from pdfplumber about invalid float values
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# --- SSL/TLS Configuration for requests ---
# This is a workaround for environments with outdated SSL libraries
# that might cause handshake failures.

# Suppress only the single InsecureRequestWarning from urllib3 needed to disable certificate checks
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class TlsAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2
        )

try:
    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
except AttributeError:
    pass

session = requests.Session()
session.mount('https://', TlsAdapter())
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})



# --- Rate Limiter ---

class RateLimiter:
    """A simple thread-safe rate limiter."""
    def __init__(self, rate_limit: int, time_window: int = 60):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """Acquire a token, blocking if necessary."""
        with self.lock:
            while self.tokens < 1:
                self._regenerate_tokens()
                time.sleep(1)
            self.tokens -= 1

    def _regenerate_tokens(self):
        """Regenerate tokens based on elapsed time."""
        now = time.time()
        time_passed = now - self.last_update
        if time_passed > self.time_window:
            self.tokens = self.rate_limit
            self.last_update = now
        else:
            new_tokens = (time_passed * self.rate_limit) / self.time_window
            if new_tokens > 1:
                self.tokens = min(self.rate_limit, self.tokens + int(new_tokens))
                self.last_update = now

jina_rate_limiter = RateLimiter(rate_limit=120, time_window=60)

# --- Text Processing Utilities ---

def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 1500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.
    """
    try:
        full_text = full_text[:100000]
        snippet_lower = snippet.lower()
        snippet_words = set(remove_punctuation(snippet_lower).split())

        best_sentence = None
        best_f1 = 0.2

        sentences = re.split(r'(?<=[.!?]) +', full_text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(remove_punctuation(sentence_lower).split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

# --- Content Extraction ---

def extract_pdf_text(url: str) -> str:
    """Extract text from a PDF URL."""
    try:
        response = session.get(url, timeout=10, verify=False)
        response.raise_for_status()
        text = ""
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception as e:
                    print(f"Error extracting text from a page in {url}: {e}")
                    continue # Continue to the next page
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving PDF from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF {url}: {e}")
        return None

def get_content_from_url(url: str, jina_api_key: str = None, pdf_processing_method: str = "Keyword Match (Fast)", query: str = "") -> str:
    """
    Fetches content from a URL, handling PDFs and using Jina API with rate limiting.
    """
    if url.lower().endswith('.pdf'):
        if pdf_processing_method == "Embedding Search (Accurate)":
            return process_pdf_with_embeddings(url, query)
        else: # Fallback to keyword match
            return extract_pdf_text(url)

    if jina_api_key:
        try:
            jina_rate_limiter.acquire()
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = session.get(f'https://r.jina.ai/{url}', headers=jina_headers, timeout=25, verify=False)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Jina API failed for {url}: {e}. Falling back to basic scrape.")

    try:
        response = session.get(url, timeout=10, verify=False)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup.find_all(['script', 'style', 'meta', 'link', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        return soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
    except requests.exceptions.RequestException as fallback_e:
        print(f"Fallback scraping failed for URL {url}: {fallback_e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during fallback for URL {url}: {e}")
        return None

# --- Main Search Function ---

def process_search_item(item: Dict, jina_api_key: str, use_jina_api: bool, pdf_processing_method: str, query: str) -> Dict:
    """
    Worker function to process a single search result item.
    """
    link = item.get("link")
    original_snippet = item.get("snippet", "")
    
    key_to_use = jina_api_key if use_jina_api else None
    
    # Pass the query for context, especially for PDF processing
    full_content = get_content_from_url(link, key_to_use, pdf_processing_method, query)
    
    snippet = original_snippet
    if full_content:
        # For embedding search, the returned content is already the relevant snippet
        if link.lower().endswith('.pdf') and pdf_processing_method == "Embedding Search (Accurate)":
            snippet = full_content
        # For regular pages or keyword-based PDF search, extract context around the snippet
        else:
            success, context_snippet = extract_snippet_with_context(full_content, original_snippet)
            snippet = context_snippet if success else full_content[:6000]
    
    return {"title": item.get("title"), "link": link, "snippet": snippet}

def google_search(env: Environment, query: str, num_results: int = 3, time_period: str = "Any time", use_jina_api: bool = True, search_pdfs: bool = False, pdf_processing_method: str = "Keyword Match (Fast)") -> List[Dict]:
    """
    Performs a Google search with retries and concurrently fetches content.
    """
    try:
        api_key = env.google_api_key
        cse_id = env.google_cse_id
        jina_api_key = env.jina_api_key

        if not api_key or not cse_id:
            st.error("Google API Key or CSE ID is not configured.")
            return []
        
        if use_jina_api and not jina_api_key:
            st.warning("Jina API is enabled, but the key is not configured. Content extraction quality may be lower.")

        service = build("customsearch", "v1", developerKey=api_key)
        
        # --- Date Restriction Logic ---
        date_restrict = None
        # Check for custom date range format YYYYMMDD..YYYYMMDD
        if re.match(r"\d{8}..\d{8}", time_period):
            date_restrict = time_period
        else:
            date_restrict_map = {
                "Past year": "y1",
                "Past 6 months": "m6",
                "Past month": "m1",
                "Past week": "w1" # Added past week for consistency
            }
            date_restrict = date_restrict_map.get(time_period)
        
        search_kwargs = {'q': query, 'cx': cse_id, 'num': num_results}
        if date_restrict:
            search_kwargs['dateRestrict'] = date_restrict

        items = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = service.cse().list(**search_kwargs).execute()
                items = res.get('items', [])
                break
            except Exception as e:
                print(f"Google Search API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e

        if not items:
            st.warning(f"No results found for: {query}")
            return []

        # Conditionally filter out PDF links if the option is disabled
        if not search_pdfs:
            items = [item for item in items if not item.get('link', '').lower().endswith('.pdf')]

        formatted_results = []
        # Use a single thread for embedding processing to avoid overwhelming memory/CPU
        max_workers = 1 if pdf_processing_method == "Embedding Search (Accurate)" else 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Pass the main query to the worker for context
            future_to_item = {executor.submit(process_search_item, item, jina_api_key, use_jina_api, pdf_processing_method, query): item for item in items}
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    formatted_results.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Error processing item {item.get('link')}: {exc}")
                    formatted_results.append({
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": f"Error processing URL: {exc}"
                    })
        
        return formatted_results

    except Exception as e:
        st.error(f"An error occurred during Google search for '{query}': {e}")
        traceback.print_exc()
        return [{"error": str(e)}]