from datetime import datetime, timedelta
from typing import Dict, List

import arxiv
from langchain.tools import Tool


class ArxivSearchTool:
    def __init__(self):
        self.max_results = 10
        self.client = arxiv.Client()

    def search_papers(
        self, query: str, max_results: int = None, days_back: int = 365
    ) -> List[Dict]:
        max_results = max_results or self.max_results

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            results = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for result in self.client.results(search):
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
            return [{"error": str(e)}]

    def as_langchain_tool(self) -> Tool:
        return Tool(
            name="arxiv_search",
            func=lambda q: self.search_papers(q),
            description=(
                "Search recent academic papers on battery technology, "
                "materials science, electrochemistry, and energy storage. "
                "Returns: title, authors, summary, PDF link."
            ),
        )
