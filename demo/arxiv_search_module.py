# arxiv_search_module.py
import streamlit as st
import arxiv
from datetime import datetime, timedelta
from typing import Dict, List

def arxiv_search(query: str, max_results: int = 3, days_back: int = 365) -> List[Dict]:
    """
    Performs a search on arXiv and returns a list of papers.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for result in client.results(search):
            # The arxiv library returns timezone-aware datetime objects.
            # We need to make our cutoff_date timezone-aware as well for comparison,
            # or make the result's published date naive.
            # Let's make the result's date naive by removing tzinfo.
            if result.published.replace(tzinfo=None) < cutoff_date:
                continue

            results.append(
                {
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary[:300] + "...",
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                }
            )
        return results
    except Exception as e:
        st.error(f"An error occurred during ArXiv search for '{query}': {e}")
        return [{"error": str(e)}]