"""Market data with fallbacks, shared by all tools.

Prices: Yahoo Finance (yfinance) first, then Alpaca's free market data (IEX feed,
needs the same ALPACA_API_KEY / ALPACA_SECRET_KEY as paper trading).
News:   Yahoo Finance first, then Google News RSS (no key needed).

Yahoo often rate-limits shared cloud IPs, so the fallbacks keep the hosted demo working.
"""
import email.utils
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import pandas as pd
import requests
import yfinance as yf

from core.logger import setup_logger

logger = setup_logger()

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PERIOD_DAYS = {"5d": 7, "1mo": 31, "2mo": 62, "3mo": 92, "6mo": 183, "1y": 366}


def _yahoo_history(symbol: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[PRICE_COLUMNS]


def _alpaca_history(symbol: str, days: int) -> pd.DataFrame:
    api_key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        return pd.DataFrame()

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=days),
        feed=DataFeed.IEX,  # the free feed
    )
    bars = client.get_stock_bars(request).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    bars = bars.rename(columns=str.capitalize)
    return bars[PRICE_COLUMNS]


def get_history(symbol: str, period: str = "3mo") -> pd.DataFrame:
    """Daily OHLCV candles for the period, or an empty DataFrame if no source has data."""
    symbol = symbol.upper().strip()
    try:
        df = _yahoo_history(symbol, period)
        if not df.empty:
            return df
        logger.warning(f"Yahoo returned no prices for {symbol}, trying Alpaca")
    except Exception as e:
        logger.warning(f"Yahoo prices failed for {symbol} ({e}), trying Alpaca")

    try:
        df = _alpaca_history(symbol, PERIOD_DAYS.get(period, 92))
        if not df.empty:
            logger.info(f"Using Alpaca prices for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Alpaca prices failed for {symbol}: {e}")
        return pd.DataFrame()


def is_valid_symbol(symbol: str) -> bool:
    return not get_history(symbol, "5d").empty


def _yahoo_news(symbol: str) -> list[dict]:
    items = []
    for raw in yf.Ticker(symbol).news or []:
        item = raw.get("content", raw)
        published = datetime.fromisoformat(item.get("pubDate")).replace(tzinfo=None)
        items.append({
            "title": item.get("title", "No Title"),
            "summary": item.get("summary", ""),
            "publisher": (item.get("provider") or {}).get("displayName", "Unknown"),
            "date": published,
            "link": (item.get("canonicalUrl") or {}).get("url", ""),
        })
    return items


def _google_news(symbol: str) -> list[dict]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(f"{symbol} stock")
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (portfolio demo)"})
    response.raise_for_status()
    items = []
    for node in ET.fromstring(response.content).findall("./channel/item"):
        published = email.utils.parsedate_to_datetime(node.findtext("pubDate"))
        source = node.find("source")
        items.append({
            "title": node.findtext("title", "No Title"),
            "summary": "",
            "publisher": source.text if source is not None else "Google News",
            "date": published.astimezone(timezone.utc).replace(tzinfo=None),
            "link": node.findtext("link", ""),
        })
    return items


def get_news(symbol: str, days_back: int = 4, limit: int = 15) -> list[dict]:
    """Recent headlines (newest first) from the last `days_back` days."""
    symbol = symbol.upper().strip()
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_back)

    items: list[dict] = []
    try:
        items = [i for i in _yahoo_news(symbol) if i["date"] > cutoff]
    except Exception as e:
        logger.warning(f"Yahoo news failed for {symbol} ({e}), trying Google News")

    if not items:
        try:
            items = [i for i in _google_news(symbol) if i["date"] > cutoff]
            logger.info(f"Using Google News for {symbol}")
        except Exception as e:
            logger.error(f"Google News failed for {symbol}: {e}")

    items.sort(key=lambda i: i["date"], reverse=True)
    return items[:limit]
