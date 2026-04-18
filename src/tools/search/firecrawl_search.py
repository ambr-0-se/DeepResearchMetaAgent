import os
import time

from dotenv import load_dotenv
load_dotenv(verbose=True)

from typing import List
from firecrawl import FirecrawlApp
import asyncio

from src.tools.search.base import WebSearchEngine, SearchItem

def search(params):
    """
    Perform a Google search using the provided parameters.
    Returns a list of SearchItem objects.
    """
    app = FirecrawlApp(
        api_key=os.getenv("FIRECRAWL_API_KEY"),
    )

    # Firecrawl-py 4.22+ rejects an empty `tbs=""` with ValueError. Only
    # include the kwarg when a non-empty filter is set.
    search_kwargs = {
        "query": params["q"],
        "limit": params.get("num", 10),
    }
    tbs = params.get("tbs")
    if tbs:
        search_kwargs["tbs"] = tbs

    response = app.search(**search_kwargs)

    # Firecrawl-py 4.22+ returns a pydantic `SearchData(web=[...], news=...,
    # images=...)` instead of the legacy `{data: [...]}` dict-shape. Support
    # both so this file keeps working across SDK upgrades.
    if hasattr(response, "web"):
        # New pydantic shape — items are SearchResultWeb models with
        # attributes (url / title / description / category).
        items = list(response.web or [])
        news = getattr(response, "news", None) or []
        items.extend(news)

        def _get(item, key, default=""):
            val = getattr(item, key, default)
            return val if val is not None else default
    else:
        # Legacy shape — list of dicts.
        items = getattr(response, "data", None) or []

        def _get(item, key, default=""):
            return item.get(key, default)

    results = []
    for item in items:
        title = _get(item, "title")
        url = _get(item, "url")
        description = _get(item, "description")
        results.append(SearchItem(title=title,
                                  url=url,
                                  description=description))

    return results

class FirecrawlSearchEngine(WebSearchEngine):
    async def perform_search(
        self,
        query: str,
        num_results: int = 10,
        filter_year: int = None,
        *args, **kwargs
    ) -> List[SearchItem]:
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        params = {
            "q": query,
            "num": num_results,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        results = search(params)

        return results


if __name__ == '__main__':
    # Example usage
    start_time = time.time()
    search_engine = FirecrawlSearchEngine()
    query = "OpenAI GPT-4"
    results = asyncio.run(search_engine.perform_search(query, num_results=5))

    for item in results:
        print(f"Title: {item.title}\nURL: {item.url}\nDescription: {item.description}\n")

    end_time = time.time()

    print(end_time - start_time, "seconds elapsed for search")