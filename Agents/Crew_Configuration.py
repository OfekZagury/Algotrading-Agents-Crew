from crewai import Agent
from langchain.llms import Ollama
# from langchain_openai import ChatOpenAI

# Psuedo-tools imports - will be replaced with actual implementations
from tools.news_scraper import NewsScraperTool
from tools.market_data import TechnicalAnalysisTool
from tools.calculator_tool import CalculatorTool
from tools.alpaca_execution import AlpacaExecutionTool

class TradingAgents:
    def __init__(self):
        # self.llm = Ollama(model="llama3")
        self.llm = None

    def sentiment_analyst_agent(self):
        return Agent(
            role='Senior Market News Analyst',
            goal='Analyze market news and sentiment for specific stocks to determine market mood.',
            backstory="""You are a veteran financial journalist with 20 years of experience.
            You don't just read headlines; you understand the nuance behind them.
            You know that "buy the rumor, sell the news" is real.
            Your job is to scrape news, filter out the noise, and provide a clear
            sentiment score (Positive/Negative/Neutral) with a brief summary.""",
            verbose=True,
            tools=[NewsScraperTool()], # Pseudo-tools, will be replaced with actual implementations
            llm=self.llm,
            allow_delegation=False
        )

    def technical_strategist_agent(self):
        return Agent(
            role='Technical Indicator Specialist',
            goal='Analyze stock charts using strict RSI and MACD rules to identify entry points.',
            backstory="""You are a strict quantitative trader. You do not guess; you follow the math.
            You must analyze the data provided by the tools and output a technical analysis based STRICTLY on these rules:

            1. **RSI Strategy**:
               - A STRONG BUY signal is when RSI is below 30 (Oversold).
               - A MODERATE BUY signal is when RSI is between 30 and 45 AND rising (current RSI > previous RSI).
               - A SELL/AVOID signal is when RSI is above 70 (Overbought).

            2. **MACD Strategy**:
               - A BUY signal exists ONLY if the MACD Line crosses ABOVE the Signal Line (Bullish Crossover).
               - If the MACD Histogram is positive and increasing, the signal is stronger.
               - If the MACD Line is below the Signal Line, it is a Bearish signal (do not buy).

            Your final output must state clearly if the technical indicators suggest 'BUY', 'SELL', or 'WAIT', citing the specific values of RSI and MACD that led to this decision.""",
            verbose=True,
            tools=[TechnicalAnalysisTool()], # Pseudo-tools, will be replaced with actual implementations
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
            tools=[TechnicalAnalysisTool(), CalculatorTool()], # Pseudo-tools, will be replaced with actual implementations
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
            tools=[AlpacaExecutionTool()], # Pseudo-tools, will be replaced with actual implementations
            llm=self.llm,
            allow_delegation=False
        )