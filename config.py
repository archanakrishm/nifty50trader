"""
Configuration for Nifty50 Trader Application
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Helper: read from env or Streamlit secrets ──────────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    """Try os.getenv first, then fall back to st.secrets (Streamlit Cloud)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

# ─── Market Settings ─────────────────────────────────────────────────────────
NIFTY50_SYMBOL = "^NSEI"
NIFTY50_NSE_SYMBOL = "NIFTY 50"
INDEX_CONSTITUENTS_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"

# Top Nifty50 stocks for individual analysis
NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "WIPRO.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TECHM.NS", "INDUSINDBK.NS",
    "BAJAJFINSV.NS", "JSWSTEEL.NS", "ONGC.NS", "COALINDIA.NS", "GRASIM.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS", "BRITANNIA.NS", "CIPLA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "TATACONSUM.NS",
    "BPCL.NS", "UPL.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS", "WIPRO.NS"
]

# ─── Data Settings ────────────────────────────────────────────────────────────
DEFAULT_PERIOD = "6mo"          # Default historical data period
DEFAULT_INTERVAL = "1d"          # Default candlestick interval
INTRADAY_INTERVAL = "5m"        # Intraday interval
LIVE_REFRESH_SECONDS = 30       # Live data refresh rate

# ─── Technical Indicator Settings ─────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
STOCH_PERIOD = 14
STOCH_SMOOTH = 3
ADX_PERIOD = 14
VWAP_PERIOD = 14
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# ─── Strategy Settings ───────────────────────────────────────────────────────
RISK_PER_TRADE = 0.02           # 2% risk per trade
MAX_OPEN_POSITIONS = 5
STOP_LOSS_PCT = 0.03            # 3% stop loss
TAKE_PROFIT_PCT = 0.06          # 6% take profit (2:1 R:R)
TRAILING_STOP_PCT = 0.02        # 2% trailing stop

# ─── Broker Settings (Zerodha) ───────────────────────────────────────────────
ZERODHA_API_KEY = _get_secret("ZERODHA_API_KEY")
ZERODHA_API_SECRET = _get_secret("ZERODHA_API_SECRET")
ZERODHA_ACCESS_TOKEN = _get_secret("ZERODHA_ACCESS_TOKEN")
ZERODHA_USER_ID = _get_secret("ZERODHA_USER_ID")

# ─── Broker Settings (Angel One / Groww) ──────────────────────────────────────
ANGEL_API_KEY = _get_secret("ANGEL_API_KEY")
ANGEL_CLIENT_ID = _get_secret("ANGEL_CLIENT_ID")
ANGEL_PASSWORD = _get_secret("ANGEL_PASSWORD")
ANGEL_TOTP_SECRET = _get_secret("ANGEL_TOTP_SECRET")

# ─── Groww Settings ──────────────────────────────────────────────────────────
GROWW_APP_ID = _get_secret("GROWW_APP_ID")
GROWW_APP_SECRET = _get_secret("GROWW_APP_SECRET")

# ─── Supabase Settings ───────────────────────────────────────────────────────
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")           # anon/public key
SUPABASE_SERVICE_KEY = _get_secret("SUPABASE_SERVICE_KEY")  # service role key

# ─── Claude AI Settings ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ─── ML Model Settings ───────────────────────────────────────────────────────
ML_LOOKBACK_DAYS = 60
ML_TRAIN_TEST_SPLIT = 0.8
ML_FEATURES = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "bb_upper", "bb_lower", "bb_mid",
    "atr", "adx", "obv", "volume_sma",
    "stoch_k", "stoch_d"
]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = _get_secret("LOG_LEVEL", "INFO")
LOG_FILE = "nifty50_trader.log"

# ─── Market Hours (IST) ──────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
