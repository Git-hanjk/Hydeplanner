# finance_search_module.py
import streamlit as st
from yfinance_search import get_financial_statements

def finance_search(query: str):
    """
    Performs a search for financial statements using a company ticker.
    
    Args:
        query (str): The company ticker symbol (e.g., "MSFT", "AAPL").

    Returns:
        dict: A dictionary containing the financial statements.
    """
    try:
        # The query for finance search is expected to be a ticker symbol.
        results = get_financial_statements(query)
        return results
    except Exception as e:
        st.error(f"An error occurred during finance search for '{query}': {e}")
        return [{"error": str(e)}]