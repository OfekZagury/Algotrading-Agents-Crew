import yfinance as yf
from datetime import datetime
from diskcache import Cache
from crewai.tools import tool
from core.logger import setup_logger
from typing import Annotated
from pydantic import Field

logger = setup_logger()



# Initialize file-based cache
cache = Cache(".cache", size_limit=int(1e9), eviction_policy='least-recently-used')

@tool("Find Important Support Level (30 Days)")
def find_support_level(
                symbol: Annotated[str, Field(description="The stock ticker (e.g., 'PLTR')")],
    ) -> float:
    """
    Identifies the significant support level based on the lowest low of the last 30 trading days.
    Returns:
        float: The support price level (rounded to 2 decimals).
    """
    # Create cache key
    cache_key = f"support_{symbol}_{datetime.now().strftime('%Y-%m-%d_%H')}"

    if cache_key in cache:
        return cache[cache_key]

    logger.info(f"Calculating support level for {symbol} (Last 30 Days)...")

    try:
        # We fetch 2 months to ensure we have at least 30 trading days
        df = yf.Ticker(symbol).history(period="2mo")

        if df.empty:
            logger.error(f"No historical data found for {symbol}")
            return 0.0

        # Take exactly the last 30 candles
        last_30_days = df.tail(30)

        # Validate we have sufficient data points
        if len(last_30_days) < 20:
            logger.warning(f"Only {len(last_30_days)} days available for support calculation (recommended: 30)")

        # Find the absolute lowest price reached during this period
        support_level = last_30_days['Low'].min()

        result = round(float(support_level), 2)

        cache[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Error finding support for {symbol}: {e}")
        return 0.0


@tool("Calculate Risk Parameters (1:3 R/R)", result_as_answer=True)
def calculate_risk_parameters(
        symbol: Annotated[str, Field(description="The stock ticker (e.g., 'PLTR')")],
        current_price: Annotated[float, Field(description="The current market price.")],
        support_level: Annotated[float, Field(description="The identified support level.")]
    ) -> dict:
    """
    Calculates precise trade parameters based on the 1:3 Risk/Reward rule.
    Logic:
    1. Stop Loss = Support Level * 0.98 (2% buffer below support).
    2. Risk = Current Price - Stop Loss.
    3. Take Profit = Current Price + (Risk * 3).
    Returns:
        dict: Containing 'entry', 'stop_loss', 'take_profit' and 'risk_per_share'.
              Returns an error dict if the trade is invalid (e.g. price below stop).
    """

    try:
        # 1. Calculate Stop Loss with a 2% buffer below support
        stop_loss = support_level * 0.98

        # 2. Calculate the Risk per share
        risk_per_share = current_price - stop_loss


        # 3. Calculate Take Profit (1:3 Ratio)
        take_profit = current_price + (risk_per_share * 3)

        result = {
            "is_valid": True,
            "symbol": symbol,
            "entry_price": round(current_price, 2),
            "stop_loss_price": round(stop_loss, 2),
            "take_profit_price": round(take_profit, 2),
            "risk_per_share": round(risk_per_share, 2),
            "risk_reward_ratio": "1:3",
            "qty": (20,000 / current_price)
        }
        return result

    except Exception as e:
        return {"error": str(e), "is_valid": False}