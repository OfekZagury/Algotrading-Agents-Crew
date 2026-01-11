import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from crewai.tools import tool
from core.logger import setup_logger
from typing import Annotated
from pydantic import Field

logger = setup_logger()

# Initialize the client (Assumption: keys are in .env)
# Note: paper=True is critical for paper trading!
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    logger.critical("Alpaca credentials not found. Paper trading disabled.")
    logger.critical("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    trading_client = None
else:
    trading_client = TradingClient(api_key, secret_key, paper=True)
    logger.info("Alpaca trading client initialized successfully (Paper Trading Mode)")

@tool("Checks the available cash (buying power) in the Alpaca Paper account")
def get_buying_power() -> float:
    """
    Checks the available cash (buying power) in the Alpaca Paper account.

    Returns:
        float: The amount of cash available for trading.
    """
    if not trading_client:
        logger.error("Trading client not initialized - check API keys")
        return 0.0

    try:
        account = trading_client.get_account()
        # Buying power for day trading is usually 4x cash, but we look at 'cash' or 'regt_buying_power'
        buying_power = float(account.buying_power)
        logger.info(f"Current buying power: ${buying_power:,.2f}")
        return buying_power
    except (ValueError, AttributeError) as e:
        logger.error(f"Invalid buying power data: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        return 0.0


@tool("Submits a BRACKET ORDER to Alpaca with Take Profit and Stop Loss")
def submit_bracket_order(
        symbol: Annotated[str, Field(description="The stock ticker to buy (e.g., 'AAPL').")],
        take_profit_price: Annotated[float, Field(description="The price to set for the Take Profit order.")],
        stop_loss_price: Annotated[float, Field(description="The price to set for the Stop Loss order.")],
        qty: Annotated[int, Field(description="Number of shares to buy (default is 1 for safety).")] = 1
    ) -> dict:
    """
    Submits a complex BRACKET ORDER to Alpaca.
    It buys the stock at MARKET price, and immediately attaches:
    1. A Take Profit Limit Order.
    2. A Stop Loss Stop Order.
    Returns:
        dict: The order details including ID and Status.
    """
    if not trading_client:
        return {"status": "error", "message": "Alpaca client not initialized (missing keys)."}

    # Validate price levels
    if stop_loss_price <= 0 or take_profit_price <= 0:
        return {"status": "error", "message": "Invalid price levels: Prices must be positive"}

    if stop_loss_price >= take_profit_price:
        return {"status": "error", "message": "Invalid setup: Stop loss must be below take profit"}

    logger.info(f"Executing BRACKET ORDER for {symbol} | Qty: {qty}")
    logger.info(f"SL: ${stop_loss_price:.2f} | TP: ${take_profit_price:.2f}")

    try:
        # Construct Take Profit Request
        tp_req = TakeProfitRequest(limit_price=take_profit_price)

        # Construct Stop Loss Request
        sl_req = StopLossRequest(stop_price=stop_loss_price)

        # Construct the Main Market Order with attached protections
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,  # Order valid for the day only
            order_class=OrderClass.BRACKET,  # Defines this as a complex bracket order
            take_profit=tp_req,
            stop_loss=sl_req
        )

        # Submit the order to Alpaca
        order = trading_client.submit_order(order_data=market_order_data)

        result = {
            "status": "success",
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": str(order.side),
            "type": "BRACKET_MARKET",
            "quantity": qty,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "filled_qty": order.filled_qty  # Note: Likely 0 at moment of submission
        }

        logger.info(f"✅ Order submitted successfully! Symbol: {symbol} | ID: {order.id}")
        return result

    except Exception as e:
        error_msg = f"Order failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}