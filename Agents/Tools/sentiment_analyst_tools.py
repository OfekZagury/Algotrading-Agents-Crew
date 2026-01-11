import os
from datetime import datetime, timedelta
from time import sleep
import yfinance as yf
from serpapi import GoogleSearch
from diskcache import Cache
from crewai.tools import tool
from core.logger import setup_logger
from typing import Annotated
from datetime import datetime

logger = setup_logger()

# Initialize file-based cache
# The .cache directory will be created automatically to store results
cache = Cache(".cache", size_limit=int(1e9), eviction_policy='least-recently-used')

# @tool("fetch_stock_news")
def fetch_stock_news(
                symbol: Annotated[str, "The stock ticker (e.g., 'PLTR')"],
                days_back: Annotated[int, "How many days back to filter news"] = 4
    ) -> list[str]:
    """
    Fetches news headlines for a given stock symbol using yfinance.
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
        ticker = yf.Ticker(symbol)
        news_items = ticker.news

        filtered_news = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for item in news_items:
            item = item['content']
            # yfinance returns timestamp in unix format
            publish_time = item.get('pubDate', 0)
            publish_date_with_tz = datetime.fromisoformat(publish_time)
            publish_date = publish_date_with_tz.replace(tzinfo=None)

            if publish_date > cutoff_date:
                title = item.get('title', 'No Title')
                publisher = item['provider'].get('displayName', 'Unknown')
                link = item['canonicalUrl'].get('url', '')
                summary = item.get('summary', 'No Summary')

                # Format the string for easier LLM readability
                formatted_item = (
                    f"--- NEWS ITEM ---\n"
                    f"Title: {title}\n"
                    f"summary: {summary}\n"
                    f"Publisher: {publisher}\n"
                    f"Date: {publish_date.strftime('%Y-%m-%d %H:%M')}\n"
                    f"Link: {link}\n"
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