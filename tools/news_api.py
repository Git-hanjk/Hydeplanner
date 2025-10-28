from datetime import datetime, timedelta
from typing import Dict, List

from langchain.tools import Tool
from newsapi import NewsApiClient


class NewsSearchTool:
    """News API tool for competitor intelligence"""

    def __init__(self, api_key: str):
        self.client = NewsApiClient(api_key=api_key)

    def search_news(
        self, query: str, days_back: int = 30, language: str = "en"
    ) -> List[Dict]:
        """Search recent news articles"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            response = self.client.get_everything(
                q=query,
                from_param=from_date,
                language=language,
                sort_by="relevancy",
                page_size=10,
            )

            articles = response.get("articles", [])
            return [
                {
                    "title": a["title"],
                    "description": a.get("description", ""),
                    "source": a["source"]["name"],
                    "url": a["url"],
                    "published_at": a["publishedAt"],
                }
                for a in articles
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="news_search",
            func=lambda q: self.search_news(q),
            description=(
                "Search recent news articles about battery technology, "
                "competitors (CATL, BYD, Samsung SDI, Panasonic), "
                "market trends, and industry developments."
            ),
        )
