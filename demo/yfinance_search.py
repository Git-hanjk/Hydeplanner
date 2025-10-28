# yfinance_search.py
import yfinance as yf
import streamlit as st

def yfinance_news_search(query: str, max_results: int = 5):
    """
    Searches for news related to a company ticker using yfinance.
    
    Args:
        query (str): The company ticker symbol (e.g., "MSFT", "AAPL").
        max_results (int): The maximum number of news articles to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a news article.
              Returns an empty list if the ticker is not found or no news is available.
    """
    try:
        ticker = yf.Ticker(query)
        # .news returns a list of dictionaries
        news = ticker.news
        
        if not news:
            st.warning(f"No news found for ticker: {query}")
            return []
            
        # Ensure the output format is consistent with other search tools
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
        st.error(f"An error occurred during yfinance search for '{query}': {e}")
        return []

if __name__ == '__main__':
    # Example usage for testing
    st.title("YFinance News Search Test")
    test_query = st.text_input("Enter a ticker (e.g., AAPL):", "AAPL")
    if st.button("Search"):
        results = yfinance_news_search(test_query)
        if results:
            st.json(results)
        else:
            st.write("No results found.")
