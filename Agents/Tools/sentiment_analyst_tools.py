import os
from datetime import datetime
from time import sleep
from Agents.Tools.market_data import get_news
from diskcache import Cache
from crewai.tools import tool
from core.logger import setup_logger
from typing import Annotated
from pydantic import Field
from datetime import datetime

logger = setup_logger()

# Initialize file-based cache
# The .cache directory will be created automatically to store results
cache = Cache(".cache", size_limit=int(1e9), eviction_policy='least-recently-used')

@tool("fetch_stock_news")
def fetch_stock_news(
                symbol: Annotated[str, Field(description="The stock ticker (e.g., 'PLTR')")],
                days_back: Annotated[int, Field(description="How many days back to filter news")] = 4
    ) -> list[str]:
    """
    Fetches recent news headlines for a stock (Yahoo Finance, falling back to Google News).
    Returns:
        list[str]: A list of formatted news strings ready for the Agent to read.
    """
    # Create a unique cache key based on symbol, time window, and current hour
    cache_key = f"news_{symbol}_{days_back}_{datetime.now().strftime('%Y-%m-%d_%H')}"

    # Check if data is already in Cache
    if cache_key in cache:
        return cache[cache_key]

    logger.info(f"Fetching fresh news for {symbol}...")

    try:
        filtered_news = []
        for item in get_news(symbol, days_back=days_back):
            # Format the string for easier LLM readability
            formatted_item = (
                f"--- NEWS ITEM ---\n"
                f"Title: {item['title']}\n"
                f"summary: {item['summary'] or 'No Summary'}\n"
                f"Publisher: {item['publisher']}\n"
                f"Date: {item['date'].strftime('%Y-%m-%d %H:%M')}\n"
                f"Link: {item['link']}\n"
            )
            filtered_news.append(formatted_item)

        # If no news found, return a generic message
        if not filtered_news:
            result = [f"No significant news found for {symbol} in the last {days_back} days."]
        else:
            result = filtered_news

        # Save result to Cache
        cache[cache_key] = result
        return result

    except Exception as e:
        error_msg = [f"Error fetching news for {symbol}: {str(e)}"]
        return error_msg