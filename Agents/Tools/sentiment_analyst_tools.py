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
            publish_date = datetime.fromisoformat(publish_time)

            if publish_date > cutoff_date:
                title = item.get('title', 'No Title')
                publisher = item['provider'].get('displayName', 'Unknown')
                link = item['canonicalUrl'].get('url', '')

                # Format the string for easier LLM readability
                formatted_item = (
                    f"--- NEWS ITEM ---\n"
                    f"Title: {title}\n"
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


# @tool("get_google_trends")
def get_google_trends(
                symbol: Annotated[str, "The stock ticker (e.g., 'PLTR')"],
    ) -> int:
    """
    Checks the Google Trends 'Hype' score (0-100) for a stock symbol via SerpApi.
    Returns:
        int: A score between 0 (no interest) and 100 (peak popularity).
    """
    # Cache key valid for the current day (trends don't change every minute)
    cache_key = f"trends_{symbol}_{datetime.now().strftime('%Y-%m-%d')}"

    if cache_key in cache:
        return cache[cache_key]

    logger.info(f"Checking Google Trends for {symbol}...")

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.error("SERPAPI_API_KEY not found in environment variables.")
        return -1  # Return -1 to indicate missing API key, not zero interest

    try:
        # Append "stock" to the query to filter out non-financial noise (e.g., Netflix the movie vs stock)
        query = f"{symbol} stock"

        params = {
            "engine": "google_trends",
            "q": query,
            "data_type": "TIMESERIES",
            "date": "now 7-d",  # Check the last 7 days
            "api_key": api_key
        }

        search = GoogleSearch(params)

        # Retry logic for network failures
        results = None
        for attempt in range(3):
            try:
                results = search.get_dict()
                break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"SerpAPI failed after 3 attempts: {e}")
                    return 0
                logger.warning(f"SerpAPI attempt {attempt + 1} failed, retrying...")
                sleep(1)

        # Extract data from the complex JSON response
        interest_over_time = results.get("interest_over_time", {})
        timeline_data = interest_over_time.get("timeline_data", [])

        if not timeline_data:
            cache[cache_key] = 0
            return 0

        # Get the most recent data point (the last item in the list)
        latest_data_point = timeline_data[-1]
        values = latest_data_point.get("values", [])

        hype_score = int(values[0]) if values else 0

        # Save to Cache
        cache[cache_key] = hype_score
        return hype_score

    except Exception as e:
        logger.error(f"Error fetching trends for {symbol}: {e}")
        return 0


if __name__ == "__main__":
    # Simple test cases to verify tool functionality
    test_symbol = "PLTR"

    news = fetch_stock_news(test_symbol, days_back=7)
    print(f"News for {test_symbol}:\n", "\n".join(news))

    trend_score = get_google_trends(test_symbol)
    print(f"Google Trends Hype Score for {test_symbol}: {trend_score}")