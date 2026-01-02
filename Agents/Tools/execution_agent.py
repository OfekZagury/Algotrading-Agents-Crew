import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# Initialize the client (Assumption: keys are in .env)
# Note: paper=True is critical for paper trading!
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

trading_client = None
if api_key and secret_key:
    trading_client = TradingClient(api_key, secret_key, paper=True)


def get_buying_power() -> float:
    """
    Checks the available cash (buying power) in the Alpaca Paper account.

    Returns:
        float: The amount of cash available for trading.
    """
    if not trading_client:
        return 0.0

    try:
        account = trading_client.get_account()
        # Buying power for day trading is usually 4x cash, but we look at 'cash' or 'regt_buying_power'
        return float(account.buying_power)
    except Exception as e:
        print(f"Error fetching account info: {e}")
        return 0.0


def submit_bracket_order(symbol: str, take_profit_price: float, stop_loss_price: float, qty: int = 1) -> dict:
    """
    Submits a complex BRACKET ORDER to Alpaca.
    It buys the stock at MARKET price, and immediately attaches:
    1. A Take Profit Limit Order.
    2. A Stop Loss Stop Order.

    Args:
        symbol (str): The ticker to buy.
        take_profit_price (float): The price to sell at profit.
        stop_loss_price (float): The price to sell at loss.
        qty (int): Number of shares to buy (Default is 1 for safety).

    Returns:
        dict: The order details including ID and Status.
    """
    if not trading_client:
        return {"status": "error", "message": "Alpaca client not initialized (missing keys)."}

    print(f"Executing BRACKET ORDER for {symbol} | Qty: {qty}")
    print(f"SL: {stop_loss_price} | TP: {take_profit_price}")

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
            "filled_qty": order.filled_qty  # Note: Likely 0 at moment of submission
        }

        print(f"✅ Order submitted successfully! ID: {order.id}")
        return result

    except Exception as e:
        error_msg = f"Order failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}