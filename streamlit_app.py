"""Live web demo: run the four-agent trading crew for a stock and watch every agent work.

Runs on Streamlit Community Cloud. The demo always simulates execution (DRY_RUN):
the crew decides and builds the bracket order, but nothing is sent to a broker.
"""
import ast
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime, timezone

import streamlit as st

# ---------------------------------------------------------------- configuration
# Hosted secrets become environment variables before the crew modules are imported.
try:
    for _key in ("GEMINI_API_KEY", "GEMINI_MODEL", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # no secrets file when running locally - .env is used instead
    pass

os.environ["DRY_RUN"] = "true"  # the public demo never places orders
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from Agents.agent_trader import build_crew  # noqa: E402
from Agents.Tools.market_data import get_history  # noqa: E402
from core.logger import setup_logger  # noqa: E402

logger = setup_logger()

# CrewAI prints emoji to the console; keep Windows consoles from choking on them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SYMBOLS = ["NFLX", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "PLTR"]
MAX_RUNS_PER_VISITOR = 3   # per browser session
MAX_RUNS_PER_DAY = 40      # across all visitors, keeps the API budget predictable
RESULT_TTL_SECONDS = 3600  # a symbol's result is reused for an hour

AGENTS = [
    ("Senior Market News Analyst", "News sentiment", "📰"),
    ("Technical Indicator Specialist", "RSI + MACD signal", "📈"),
    ("Risk & Strategy Manager", "Risk plan (1:3)", "🛡️"),
    ("Head of Execution", "Bracket order (simulated)", "⚡"),
]

st.set_page_config(page_title="Algotrading Agents Crew · Live demo", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- shared state
@st.cache_resource
def shared_store():
    """Process-wide store shared by all visitors: cached results and the daily run counter."""
    return {"results": {}, "day": None, "runs_today": 0, "lock": threading.Lock()}


def cached_result(symbol):
    entry = shared_store()["results"].get(symbol)
    if entry and time.time() - entry["at"] < RESULT_TTL_SECONDS:
        return entry
    return None


def reserve_daily_run():
    store = shared_store()
    with store["lock"]:
        today = datetime.now(timezone.utc).date().isoformat()
        if store["day"] != today:
            store["day"], store["runs_today"] = today, 0
        if store["runs_today"] >= MAX_RUNS_PER_DAY:
            return False
        store["runs_today"] += 1
        return True


@st.cache_data(ttl=3600, show_spinner=False)
def price_history(symbol):
    return get_history(symbol, "3mo")


# ---------------------------------------------------------------- result parsing
def parse_summary(outputs):
    """Pull the key numbers out of the agents' text outputs (best effort)."""
    summary = {}
    if len(outputs) > 0:
        m = re.search(r"Sentiment:\s*\**\s*(Positive|Negative|Neutral)", outputs[0], re.I)
        summary["sentiment"] = m.group(1).capitalize() if m else None
    if len(outputs) > 1:
        # Either "DECISION: WAIT | RSI=..." text or the tool's dict {'Decision': 'WAIT', 'RSI': 33.6, ...}
        m = re.search(r"""['"]?DECISION['"]?\s*[:=]\s*\**\s*['"]?(BUY|SELL|WAIT)""", outputs[1], re.I)
        summary["decision"] = m.group(1).upper() if m else None
        m = re.search(r"""['"]?RSI['"]?\s*[:=]\s*([\d.]+)""", outputs[1])
        summary["rsi"] = m.group(1) if m else None
    if len(outputs) > 2:
        risk = as_dict(outputs[2])
        summary["risk"] = risk if isinstance(risk, dict) else None
    return summary


def as_dict(text):
    """Parse a JSON object or a Python dict literal out of an agent's output, else None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(m.group(0))
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            continue
    return None


def show_output(text):
    """Structured outputs (dicts / JSON) as a readable list, everything else as markdown."""
    data = as_dict(text) if text.strip().startswith("{") else None
    if data:
        st.markdown("\n".join(f"- **{k.replace('_', ' ')}:** {v}" for k, v in data.items()))
    else:
        st.markdown(text)


def render_summary(symbol, outputs):
    s = parse_summary(outputs)
    cols = st.columns(3)
    cols[0].metric("News sentiment", s.get("sentiment") or "–")
    cols[1].metric("Technical decision", s.get("decision") or "–", help=f"RSI {s['rsi']}" if s.get("rsi") else None)
    risk = s.get("risk") or {}
    if risk.get("is_valid"):
        cols[2].metric("Trade", f"BUY {risk.get('qty', '?')} × {symbol}")
        st.success(
            f"**Simulated bracket order** · entry ≈ ${risk.get('entry_price')} · "
            f"stop loss ${risk.get('stop_loss_price')} · take profit ${risk.get('take_profit_price')} "
            f"· risk/reward {risk.get('risk_reward_ratio', '1:3')}"
        )
    else:
        cols[2].metric("Trade", "No trade")
        reason = risk.get("reason") if isinstance(risk, dict) else None
        st.info(f"The crew decided not to trade{f': {reason}' if reason else ''}.")


def render_agent_cards(outputs, running_index=None):
    for i, (role, subtitle, icon) in enumerate(AGENTS):
        if i < len(outputs):
            with st.expander(f"{icon} **{role}** · {subtitle} ✅", expanded=(i == len(outputs) - 1)):
                show_output(outputs[i])
        elif i == running_index:
            st.info(f"{icon} **{role}** is working…")
        else:
            st.caption(f"{icon} {role} · waiting")


# ---------------------------------------------------------------- page
st.title("📈 Algotrading Agents Crew")
st.markdown(
    "Four AI agents built with **CrewAI** and powered by **Google Gemini** analyze a stock as a team: "
    "a news analyst reads recent headlines, a technical strategist computes RSI and MACD, a risk manager "
    "sets the stop loss and take profit from the 30-day support level, and an execution trader builds a "
    "bracket order."
)
st.caption(
    "Demo mode: orders are **simulated** and never sent to a broker. Educational project, not financial advice. "
    "· [Source code on GitHub](https://github.com/OfekZagury/Algotrading-Agents-Crew)"
)

if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    st.error("The demo is not configured: no GEMINI_API_KEY found.")
    st.stop()

if "runs" not in st.session_state:
    st.session_state.runs = 0

left, right = st.columns([1, 2], gap="large")
with left:
    symbol = st.selectbox("Stock", SYMBOLS, index=0)
    history = price_history(symbol)
    if not history.empty:
        last, first = history["Close"].iloc[-1], history["Close"].iloc[0]
        st.metric(f"{symbol} · last close", f"${last:,.2f}", f"{(last / first - 1) * 100:+.1f}% (3 months)")
        st.line_chart(history["Close"], height=220)
    else:
        st.warning("Price data is temporarily unavailable for this symbol.")

    cached = cached_result(symbol)
    visitor_left = MAX_RUNS_PER_VISITOR - st.session_state.runs
    run_clicked = st.button(
        "▶ Run the crew",
        type="primary",
        use_container_width=True,
        disabled=cached is not None or visitor_left <= 0,
    )
    if cached:
        minutes = int((time.time() - cached["at"]) // 60)
        st.caption(f"Showing the {symbol} run from {minutes} min ago (results are reused for an hour to keep the demo free).")
    elif visitor_left <= 0:
        st.caption("You've used this session's runs - try another time, or check the cached results of other stocks.")
    else:
        st.caption(f"A run takes about a minute. Runs left in this session: {visitor_left}.")

with right:
    # One slot for the whole column, so a new render replaces the previous one
    # instead of leaving stale (greyed-out) elements underneath.
    panel = st.empty()

    if cached and not run_clicked:
        with panel.container():
            render_summary(symbol, cached["outputs"])
            render_agent_cards(cached["outputs"])

    elif run_clicked:
        if not reserve_daily_run():
            panel.warning("The demo reached today's run limit. Please come back tomorrow.")
            st.stop()
        st.session_state.runs += 1

        events: queue.Queue = queue.Queue()

        def worker():
            try:
                crew = build_crew(symbol, task_callback=lambda out: events.put(("task", out.raw)), verbose=False)
                crew.kickoff()
                events.put(("done", None))
            except Exception as exc:
                events.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()

        outputs: list[str] = []
        while True:
            with panel.container():
                render_agent_cards(outputs, running_index=len(outputs))
            kind, payload = events.get()
            if kind == "task":
                outputs.append(payload)
            elif kind == "error":
                logger.error(f"Crew run for {symbol} failed: {payload!r}")
                # A failed run doesn't count against the visitor or the daily budget
                st.session_state.runs -= 1
                store = shared_store()
                with store["lock"]:
                    store["runs_today"] = max(0, store["runs_today"] - 1)
                panel.error(
                    "The AI service didn't respond this time (it may be busy or rate-limited). "
                    "Please try again in a minute."
                )
                break
            else:
                shared_store()["results"][symbol] = {"outputs": outputs, "at": time.time()}
                # Rerun so the whole page reflects the new state (cached result, runs left).
                st.rerun()
    else:
        with panel.container():
            st.info("Pick a stock and press **Run the crew** to watch the four agents work step by step.")
            render_agent_cards([])
