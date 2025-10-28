# yfinance_search.py
import yfinance as yf
import streamlit as st
import requests
import re
import json

def get_ticker_from_name(company_name: str) -> str:
    """
    Searches Yahoo Finance for a company name and returns the most likely ticker symbol.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # The 'quotes' list contains the search results.
        # The first result is often the most relevant.
        if data.get('quotes'):
            return data['quotes'][0]['symbol']
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching ticker for '{company_name}': {e}")
    except (KeyError, IndexError):
        st.warning(f"Could not find a ticker for '{company_name}'.")
    return None

def get_financial_statements(ticker_symbol: str):
    """
    Retrieves financial statements (Income Statement, Balance Sheet, Cash Flow) for a given ticker.
    
    Args:
        ticker_symbol (str): The company ticker symbol (e.g., "MSFT").

    Returns:
        dict: A dictionary containing the financial statements as JSON strings.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get financial statements
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        
        if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
            st.warning(f"No financial data found for ticker: {ticker_symbol}")
            return {}

        # Convert dataframes to JSON
        statements = {
            "income_statement": income_stmt.to_json(orient='split'),
            "balance_sheet": balance_sheet.to_json(orient='split'),
            "cash_flow": cash_flow.to_json(orient='split')
        }
        
        return statements
        
    except Exception as e:
        st.error(f"An error occurred during yfinance financial data fetch for '{ticker_symbol}': {e}")
        return {}


def yfinance_news_search(query: str, max_results: int = 5):
    """
    Searches for news related to a company name or ticker using yfinance.
    
    Args:
        query (str): The company name or ticker symbol (e.g., "Apple", "MSFT").
        max_results (int): The maximum number of news articles to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a news article.
    """
    ticker_symbol = query
    # Simple check: if the query is likely a company name (contains spaces or is longer than 5 chars),
    # try to find a ticker for it. This is a basic heuristic.
    if ' ' in query or len(query) > 5 or not query.isupper():
        st.info(f"Searching for ticker for '{query}'...")
        found_ticker = get_ticker_from_name(query)
        if found_ticker:
            st.success(f"Found ticker '{found_ticker}' for '{query}'.")
            ticker_symbol = found_ticker
        else:
            st.warning(f"Could not automatically determine ticker for '{query}'. Using original query.")
            return []

    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            st.warning(f"No news found for ticker: {ticker_symbol}")
            return []
            
        formatted_news = [
            {
                "title": article.get("title", "No Title"),
                "link": article.get("link", "No Link"),
                "publisher": article.get("publisher", "No Publisher"),
                "providerPublishTime": article.get("providerPublishTime", "No Time"),
            }
            for article in news[:max_results]
        ]
        
        return formatted_news
        
    except Exception as e:
        st.error(f"An error occurred during yfinance search for '{ticker_symbol}': {e}")
        return []

if __name__ == '__main__':
    st.title("YFinance Financials Search Test")
    test_query = st.text_input("Enter a company ticker (e.g., AAPL):", "AAPL")
    if st.button("Search Financials"):
        results = get_financial_statements(test_query)
        if results:
            st.json(results)
        else:
            st.write("No results found.")