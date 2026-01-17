from dotenv import load_dotenv
from crewai import Crew, Process, Task
from Agents.Crew_Configuration import TradingAgents
from core.logger import setup_logger

logger = setup_logger()

# Load environment variables (API keys)
load_dotenv()


def main():
    """Main function to run the Algotrading CrewAI System.
    Main Flow:
    1. Initialize Agents
    2. Define Tasks
    3. Assemble the Crew
    4. Kickoff the Crew
    5. Log Results
    6. Repeat for multiple stocks
    """
    logger.info("Starting Algotrading CrewAI System...")

    # 1. Initialize Agents
    agents_factory = TradingAgents()

    news_agent = agents_factory.sentiment_analyst_agent()
    technical_agent = agents_factory.technical_strategist_agent()
    risk_agent = agents_factory.risk_management_agent()
    execution_agent = agents_factory.execution_agent()

    # List of stocks to trade
    symbols = ['NFLX']

    for symbol in symbols:
        logger.info(f"\n\n--- Starting Analysis for {symbol} ---")

        # 2. Define Tasks
        # Task 1: Sentiment Analysis
        # This agent works independently to gather qualitative data.
        task_news = Task(
            description=f"""
                1. Use tool to fetch latest news for **{symbol}**.
                2. Analyze the sentiment (Positive/Negative/Neutral).
                3. Provide a summary of key catalysts.
            """,
            agent=news_agent,
            expected_output="Sentiment: [Positive/Negative/Neutral] | News Summary: [...] "

        )

        # Task 2: Technical Analysis
        # This agent fetches hard numbers. It doesn't strictly need the news, but runs in parallel/sequence.
        task_technical = Task(
            description=f"""
            1. Review the Sentiment Analysis from the News Agent for {symbol}.
            2. Call 'calculate RSI and MACD values and determine trend' ONCE for {symbol}.
            3. Based on the results AND sentiment:
               - RSI < 30 AND MACD Bullish AND Positive/Neutral sentiment → BUY signal
               - RSI > 70 AND MACD Bearish AND Negative/Neutral sentiment → SELL signal  
               - If sentiment strongly contradicts technical indicators, mention this conflict
               - Otherwise → WAIT signal
            4. Output format: "DECISION: [BUY/SELL/WAIT] | RSI: [value] | MACD: [Bullish/Bearish] | Price: [value] | Sentiment Alignment: [Aligned/Conflicted]"
            5. Do NOT call tools again after getting valid data.
            """,
            agent=technical_agent,
            context=[task_news],  # Incorporate sentiment to validate technical signals
            expected_output="JSON-like output with decision, RSI, MACD, Current Price, and sentiment alignment status."
        )

        # Task 3: Risk Calculation
        # CRITICAL: This task receives the output from 'task_technical' via 'context'.
        task_risk = Task(
            description=f"""
                1. Read Technical Agent's decision for {symbol}
                2. If 'SELL'/'WAIT' → Return skip JSON
                3. If 'BUY' → Call tools and return their output unchanged
                4. Output ONLY valid JSON - no markdown, no text
            """,
            agent=risk_agent,
            context=[task_technical],
            expected_output="""
            ONLY ONE of these JSON objects (no markdown, no extra text):

            Valid trade:
            {
              "is_valid": true,
              "symbol": "<TICKER>",
              "entry_price": <number>,
              "stop_loss_price": <number>,
              "take_profit_price": <number>,
              "risk_per_share": <number>,
              "risk_reward_ratio": "1:3",
              "qty": <integer>
            }

            Or skip:
            {
              "is_valid": false,
              "status": "SKIP",
              "reason": "NO TRADE OPPORTUNITY"
            }
            """
        )

        # Task 4: Execution
        # CRITICAL: This task receives the output from 'task_risk'.
        task_execution = Task(
            description=f"""
                1. Read the Risk Agent's plan for {symbol}.
                2. If the plan says "SKIP TRADE" or is invalid, do nothing.
                3. If valid, check buying power and submit the BRACKET ORDER to Alpaca.
            """,
            agent=execution_agent,
            context=[task_risk],  # <--- This links the output of Risk Agent to Execution Agent
            expected_output="Final transaction log: Order ID and status, or reason for no trade."
        )

        # 3. Assemble the Crew
        # The sequential process ensures tasks run 1 -> 2 -> 3 -> 4
        trading_crew = Crew(
            agents=[news_agent, technical_agent, risk_agent, execution_agent],
            tasks=[task_news, task_technical, task_risk, task_execution],
            process=Process.sequential,
            verbose=True
        )

        # 4. Kickoff!
        result = trading_crew.kickoff()

        logger.info(f"\nFinished cycle for {symbol}. Result:")
        logger.info(result)


if __name__ == "__main__":
    main()