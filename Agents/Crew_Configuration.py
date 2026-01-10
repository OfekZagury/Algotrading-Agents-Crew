from crewai import Agent
# from langchain.llms import Ollama
# from langchain_openai import ChatOpenAI

from Agents.Tools.sentiment_analyst_tools import fetch_stock_news, get_google_trends
from Agents.Tools.technical_strategist_tools import get_historical_data, calculate_indicators
from Agents.Tools.risk_management_tools import find_support_level, calculate_risk_parameters
from Agents.Tools.execution_agent_tools import get_buying_power, submit_bracket_order

class TradingAgents:
    def __init__(self):
        # self.llm = Ollama(model="llama3")
        self.llm = None

    def sentiment_analyst_agent(self):
        return Agent(
            role='Senior Market News Analyst',
            goal='Analyze market news and sentiment ONLY for the stocks [PLTR (palantir), NFLX (netflix), PLTK (playtika)] to determine market mood.',
            backstory="""You are an expert market analyst executing a strict two-step workflow:

            1. Use `fetch_stock_news` to identify key catalysts (earnings, upgrades, macro events) while strictly filtering out noise.
            2. Use `get_google_trends` to validate sentiment intensity. High trends amplify the news impact; low trends suggest weak market interest.
            
            Your final output must strictly follow this format:
            - Sentiment: Positive / Negative / Neutral
            - News Summary: (Key drivers found)
            - Trend Score: (Value & explanation of how it impacts your confidence)""",
            verbose=True,
            tools=[fetch_stock_news, get_google_trends],
            llm=self.llm,
            allow_delegation=False
        )

    def technical_strategist_agent(self):
        return Agent(
            role='Technical Indicator Specialist',
            goal='Analyze technical indicators from tool output to generate strict BUY/SELL/WAIT signals.',
            backstory="""You are a strict quantitative trader. You do not analyze charts visually; you analyze the JSON data returned by `calculate_indicators`.

            Apply these EXACT rules to the data:

            1. **RSI Logic** (Check 'RSI' value):
               - **STRONG BUY**: RSI <= 30.
               - **MODERATE BUY**:  30 < RSI <= 50.
               - **SELL/WAIT**: RSI > 50 (Not an attractive entry point).
               - **STRONG SELL**: RSI > 70.

            2. **MACD Logic** (Check 'Trend' and 'MACD_Line'):
               - **BUY SIGNAL**: The 'Trend' field is 'Bullish' (MACD_Line > Signal_Line).
               - **SELL SIGNAL**: The 'Trend' field is 'Bearish'.

            **Final Decision Matrix**:
            - **BUY**: Requires BOTH a Bullish MACD Trend AND RSI <= 50.
            - **WAIT**: If MACD is Bullish but RSI is too high (>50), or if MACD is Bearish.
            - **SELL**: If RSI > 70.

            Output format:
            "DECISION: [BUY/SELL/WAIT] | DATA: RSI=[Value], MACD=[Trend]" """,
            verbose=True,
            tools=[get_historical_data, calculate_indicators],
            llm=self.llm,
            allow_delegation=False
        )

    def risk_management_agent(self):
        return Agent(
            role='Risk & Strategy Manager',
            goal='Validate trade viability and retrieve strict Entry/SL/TP values using risk tools.',
            backstory="""You are a precise Risk Manager. You DO NOT perform math yourself.
            You must execute this strictly tool-based workflow:

            1. Receive the 'Current Price' from the Technical Strategist's analysis.
            2. Call `find_support_level(symbol)` to get the recent support price.
            3. Call `calculate_risk_parameters(current_price, support_level)`. This tool will return a JSON.

            4. **Analyze the Tool Output**:
               - If the tool returns `"is_valid": False`: Abort the trade immediately. Output "RISK ASSESSMENT FAILED" and the error reason.
               - If the tool returns `"is_valid": True`: Extract the precise 'entry_price', 'stop_loss', and 'take_profit'.

            Your final answer must be a clear execution order containing those three exact numbers for the next agent.""",
            verbose=True,
            tools=[find_support_level, calculate_risk_parameters],
            llm=self.llm,
            allow_delegation=False
        )

    def execution_agent(self):
        return Agent(
            role='Head of Execution',
            goal='Execute precise Bracket Orders via Alpaca based on approved Risk parameters.',
            backstory="""You are the final gatekeeper. You receive a structured execution plan from the Risk Manager.

            Protocol:
            1. **Validation**: Check if the Risk Manager returned `"is_valid": True`. If False, STOP immediately and report the error.
            2. **Funds Check**: Call `get_buying_power()`. Ensure you have enough cash to cover the `entry_price`.
            3. **Execution**: 
               - If funds are sufficient, call `submit_bracket_order`.
               - Pass the EXACT `stop_loss` and `take_profit` figures provided by the Risk Manager.
               - Note: The tool uses a Market Order for entry to ensure immediate execution.

            Your Final Output: The confirmed Alpaca `order_id` and status, or a specific reason for abortion (e.g., "Insufficient Funds").""",
            verbose=True,
            tools=[get_buying_power, submit_bracket_order],
            llm=self.llm,
            allow_delegation=False
        )