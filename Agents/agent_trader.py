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
    symbols = ['PLTR', 'NFLX', 'PLTK']

    for symbol in symbols:
        logger.info(f"\n\n--- Starting Analysis for {symbol} ---")

        # 2. Define Tasks
        # Task 1: Sentiment Analysis
        # This agent works independently to gather qualitative data.
        task_news = Task(
            description=f"""
                1. Use tools to fetch latest news and Google Trends for **{symbol}**.
                2. Analyze the sentiment (Positive/Negative/Neutral).
                3. Provide a summary of key catalysts.
            """,
            agent=news_agent,
            expected_output="Sentiment: [Positive/Negative/Neutral] | News Summary: [...] | Trend Score: [value]"

        )

        # Task 2: Technical Analysis
        # This agent fetches hard numbers. It doesn't strictly need the news, but runs in parallel/sequence.
        task_technical = Task(
            description=f"""
                1. Use the technical tool to calculate RSI, MACD, and Price for **{symbol}**.
                2. Apply the strict BUY/SELL logic defined in your backstory.
                3. Output the specific decision (BUY/SELL/WAIT) and the raw data.
            """,
            agent=technical_agent,
            context=[task_news],  # Use the news to potentially inform technical decisions
            expected_output="JSON-like output with decision, RSI, MACD, and Current Price."
        )

        # Task 3: Risk Calculation
        # CRITICAL: This task receives the output from 'task_technical' via 'context'.
        task_risk = Task(
            description=f"""
                1. Read the Technical Agent's output. 
                2. If decision is 'SELL' or 'WAIT', output "NO TRADE OPPORTUNITY" and stop.
                3. If 'BUY', use the `find_support_level` and `calculate_risk_parameters` tools.
                4. Return a validated JSON with Entry, Stop Loss, and Take Profit.
            """,
            agent=risk_agent,
            context=[task_technical],  # <--- This links the output of Technical Agent to Risk Agent
            expected_output="A JSON object with trade parameters or a skip message."
        )

        # Task 4: Execution
        # CRITICAL: This task receives the output from 'task_risk'.
        task_execution = Task(
            description=f"""
                1. Read the Risk Agent's plan.
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