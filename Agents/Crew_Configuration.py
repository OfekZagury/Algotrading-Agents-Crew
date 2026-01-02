from crewai import Agent
from langchain.llms import Ollama
# from langchain_openai import ChatOpenAI

from Agents.Tools.sentiment_analyst_tools import fetch_stock_news, get_google_trends
from Agents.Tools.technical_strategist_tools import get_historical_data, calculate_indicators

class TradingAgents:
    def __init__(self):
        # self.llm = Ollama(model="llama3")
        self.llm = None

    def sentiment_analyst_agent(self):
        return Agent(
            role='Senior Market News Analyst',
            goal='Analyze market news and sentiment for specific stocks to determine market mood.',
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
               - **STRONG BUY**: RSI < 30.
               - **MODERATE BUY**: RSI between 30 and 45.
               - **SELL/WAIT**: RSI > 45 (Not an attractive entry point).
               - **STRONG SELL**: RSI > 70.

            2. **MACD Logic** (Check 'Trend' and 'MACD_Line'):
               - **BUY SIGNAL**: The 'Trend' field is 'Bullish' (MACD_Line > Signal_Line).
               - **SELL SIGNAL**: The 'Trend' field is 'Bearish'.

            **Final Decision Matrix**:
            - **BUY**: Requires BOTH a Bullish MACD Trend AND RSI < 45.
            - **WAIT**: If MACD is Bullish but RSI is too high (>45), or if MACD is Bearish.
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
            goal='Define trade parameters: Entry, Stop Loss (2% below support), and Take Profit (1:3 ratio).',
            backstory="""You must strictly calculate trade parameters using these rules:
            1. Identify the nearest Support Level in the Daily time frame.
            2. Set Stop Loss (SL) = Support Level * 0.98 (2% buffer below support).
            3. Calculate Risk = Current Price - Stop Loss.
            4. Set Take Profit (TP) = Current Price + (Risk * 3) to ensure a 1:3 Risk/Reward ratio.

            Output the precise Entry Price, Stop Loss, and Take Profit values based on these calculations.""",
            verbose=True,
            tools=[],
            llm=self.llm,
            allow_delegation=False
        )

    def execution_agent(self):
        return Agent(
            role='Head of Execution',
            goal='Execute precise Bracket Orders (Entry + SL + TP) on Alpaca based on Risk Manager params.',
            backstory="""You receive precise 'Entry Price', 'Stop Loss', and 'Take Profit' values from the Risk Manager.
            Your execution process is strict:
            1. Check account 'Buying Power' using the Alpaca tool.
            2. If funds allow, submit a BRACKET ORDER (Limit Buy + attached Stop Loss & Take Profit).
               - Do NOT send a simple market order. You must attach the protection levels immediately.
            3. Return the executing Order ID and Status as confirmation.

            If buying power is insufficient, abort and output 'INSUFFICIENT FUNDS'.""",
            verbose=True,
            tools=[],
            llm=self.llm,
            allow_delegation=False
        )