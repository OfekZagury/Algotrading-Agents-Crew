from crewai import Agent
from langchain.llms import Ollama  # אם אתה רוצה לרוץ לוקאלית
# from langchain_openai import ChatOpenAI # אם אתה רוצה להשתמש ב-GPT

# נניח שהכלים האלו קיימים בתיקיית tools שלנו (נבנה אותם בהמשך)
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
            tools=[NewsScraperTool()],
            llm=self.llm,
            allow_delegation=False
        )

    def technical_strategist_agent(self):
        return Agent(
            role='Technical Indicator Specialist',
            goal='Analyze stock charts using RSI and MACD to identify trend reversals.',
            backstory="""You are a quantitative trader who believes "Price discounts everything".
            You don't care about news or rumors. You only care about the math.
            You look for RSI divergence, overbought/oversold levels, and MACD crossovers.
            Your output must be data-driven and precise.""",
            verbose=True,
            tools=[TechnicalAnalysisTool()],
            llm=self.llm,
            allow_delegation=False
        )

    def risk_management_agent(self):
        return Agent(
            role='Risk & Strategy Manager',
            goal='Plan the trade structure: Entry price, Stop Loss, and Take Profit.',
            backstory="""You are the responsible adult in the room. You care about capital preservation first.
            You analyze the data from the Technical Strategist to find Support and Resistance levels.
            You calculate the Risk/Reward ratio.
            You are the one who decides exactly at what price to enter and where to exit
            to ensure we don't lose more than we can afford.""",
            verbose=True,
            tools=[TechnicalAnalysisTool(), CalculatorTool()],
            llm=self.llm,
            allow_delegation=False
        )

    def execution_agent(self):
        return Agent(
            role='Head of Execution',
            goal='Execute trades on Alpaca based on the approved strategy.',
            backstory="""You are a precise execution trader.
            Your job is to take the plan from the Risk Manager and execute it in the real market.
            You check the account balance before every trade.
            You ensure orders are sent correctly to the Alpaca Paper Trading API.
            You log every action detailedly.""",
            verbose=True,
            tools=[AlpacaExecutionTool()],
            llm=self.llm,
            allow_delegation=False
        )