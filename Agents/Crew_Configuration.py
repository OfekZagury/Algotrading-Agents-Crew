import os

from crewai import Agent, LLM

from Agents.Tools.sentiment_analyst_tools import fetch_stock_news
from Agents.Tools.technical_strategist_tools import get_historical_data, calculate_indicators
from Agents.Tools.risk_management_tools import find_support_level, calculate_risk_parameters
from Agents.Tools.execution_agent_tools import get_buying_power, submit_bracket_order

class TradingAgents:
    def __init__(self):
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        # Reasoning models (gpt-5, o-series) only accept the default temperature;
        # for other models keep temperature=0 for deterministic decisions.
        is_reasoning_model = model.startswith(("gpt-5", "o1", "o3", "o4"))
        self.llm = LLM(model=model) if is_reasoning_model else LLM(model=model, temperature=0)

    def sentiment_analyst_agent(self):
        return Agent(
            role='Senior Market News Analyst',
            goal='Analyze market news and sentiment ONLY for the stock symbol given in the task to determine market mood.',
            backstory="""You are an expert market analyst executing a strict two-step workflow:

            Use `fetch_stock_news` to identify key catalysts (earnings, upgrades, macro events) while strictly filtering out noise.
            
            If tools return empty/null, output "INSUFFICIENT DATA - Cannot Assess Sentiment"
            
            Your final output must strictly follow this format:
            - Sentiment: Positive / Negative / Neutral
            - News Summary: (Key drivers found)
            """,
            verbose=True,
            tools=[fetch_stock_news],
            llm=self.llm,
            max_iter=3,
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
            max_iter=3,
            allow_delegation=False
        )

    def risk_management_agent(self):
        return Agent(
            role='Risk & Strategy Manager',
            goal='Validate trade viability and return exact risk parameters from tools.',
            backstory="""You are a Risk Manager. Follow this exact workflow:

            1. Extract Symbol and Current Price from the Technical Agent's output.
            2. If Technical decision is 'SELL' or 'WAIT', immediately return:
               {"is_valid": false, "status": "SKIP", "reason": "NO TRADE OPPORTUNITY"}

            3. If Technical decision is 'BUY':
               a. Call find_support_level(symbol)
               b. Call calculate_risk_parameters(symbol, current_price, support_level)
               c. Return the EXACT JSON output from calculate_risk_parameters

            CRITICAL: Your Final Answer must be ONLY a JSON object. 
            Do NOT include "Thought:", "Action:", or any text outside the JSON.
            Do NOT modify the tool's output - return it as-is.""",
            verbose=True,
            tools=[find_support_level, calculate_risk_parameters],
            llm=self.llm,
            max_iter=3,
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
            max_iter=3,
            allow_delegation=False
        )