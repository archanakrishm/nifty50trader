# 📈 Nifty50 Trader

A full-stack algorithmic trading application that fetches live Nifty 50 market data, performs candlestick pattern analysis, computes 15+ technical indicators, runs ML predictions, generates trading signals, and can execute trades through your demat account — all backed by **Supabase** for persistence.

---

## Features

| Module | Capabilities |
|--------|-------------|
| **Live Data** | Real-time Nifty 50 index & stock prices via yfinance + NSE APIs |
| **Candlestick Analysis** | 13 patterns detected: Doji, Hammer, Engulfing, Morning/Evening Star, Three Soldiers/Crows, and more |
| **Technical Indicators** | RSI, MACD, Bollinger Bands, SuperTrend, ADX, Ichimoku, Stochastic, OBV, VWAP, Fibonacci, Support/Resistance |
| **ML Predictions** | Random Forest + Gradient Boosting ensemble for next-day direction prediction |
| **Strategy Engine** | Multi-factor scoring system combining patterns + indicators + ML into BUY/SELL signals with confidence |
| **Risk Management** | Position sizing, stop loss, take profit, trailing stop, max position limits |
| **Broker Integration** | Zerodha (Kite Connect), Angel One (SmartAPI), Groww (via Angel One) |
| **Supabase DB** | Persists signals, trades, portfolio snapshots, watchlist, strategy performance, settings |
| **Dashboard** | Interactive Streamlit UI with charts, scanner, trade execution, and portfolio tracking |

---

## Quick Start

### 1. Install Dependencies

```bash
cd nifty50_trader
pip install -r requirements.txt
```

### 2. Set Up Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the schema:

```sql
-- Copy contents of utils/supabase_schema.sql and execute
```

3. Copy your project URL and anon key

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual keys:
```

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key

# Broker keys (at least one)
ZERODHA_API_KEY=...
ZERODHA_API_SECRET=...
ZERODHA_ACCESS_TOKEN=...

# OR Angel One / Groww
ANGEL_API_KEY=...
ANGEL_CLIENT_ID=...
ANGEL_PASSWORD=...
ANGEL_TOTP_SECRET=...
```

### 4. Run the App

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

---

## Architecture

```
nifty50_trader/
├── app.py                    # Streamlit dashboard (main entry)
├── config.py                 # All configuration & constants
├── .env.example              # Environment template
├── requirements.txt
│
├── core/
│   ├── data_fetcher.py       # Live & historical data (yfinance + NSE)
│   ├── candlestick_analyzer.py  # 13 Japanese candlestick patterns
│   ├── indicators.py         # 15+ technical indicators
│   ├── strategy.py           # Signal engine + portfolio manager
│   └── ml_predictor.py       # ML ensemble (RF + GB)
│
├── broker/
│   ├── base.py               # Abstract broker interface
│   ├── zerodha_broker.py     # Zerodha Kite Connect
│   ├── angelone_broker.py    # Angel One SmartAPI (+ Groww)
│   └── executor.py           # Signal → Order execution + DB logging
│
├── utils/
│   ├── database.py           # Supabase CRUD operations
│   └── supabase_schema.sql   # Database schema (run in Supabase SQL Editor)
│
├── models/                   # Saved ML model files (auto-generated)
└── static/                   # Static assets
```

---

## Broker Setup

### Zerodha
1. Sign up at [Kite Connect](https://kite.trade)
2. Create an app to get API key & secret
3. Generate access token via login flow
4. Set `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_ACCESS_TOKEN` in `.env`

### Angel One / Groww
Groww's demat runs on Angel One's backend, so you can use **SmartAPI**:
1. Sign up at [Angel One SmartAPI](https://smartapi.angelbroking.com)
2. Get your API key and enable TOTP
3. Set `ANGEL_API_KEY`, `ANGEL_CLIENT_ID`, `ANGEL_PASSWORD`, `ANGEL_TOTP_SECRET` in `.env`

---

## Supabase Tables

| Table | Purpose |
|-------|---------|
| `trade_signals` | Every generated signal with reasons & scores |
| `trades` | Executed orders with P&L tracking |
| `portfolio_snapshots` | Periodic capital & holdings snapshots |
| `market_data_cache` | OHLCV cache to reduce API calls |
| `watchlist` | User's watchlist |
| `strategy_performance` | Daily strategy metrics (win rate, drawdown) |
| `app_settings` | User preferences (key-value) |

---

## How the Strategy Works

1. **Fetch** OHLCV data for Nifty 50 / individual stocks
2. **Compute** 15+ technical indicators (RSI, MACD, BB, SuperTrend, ADX, etc.)
3. **Detect** candlestick patterns (Doji, Engulfing, Morning Star, etc.)
4. **Score** each factor independently (bullish = positive, bearish = negative)
5. **ML** ensemble predicts next-day direction probability
6. **Combine** all scores into a composite signal with confidence level
7. **Risk management** calculates position size, stop loss, take profit
8. **Execute** via broker API (or dry-run simulation)
9. **Persist** everything to Supabase for tracking and analytics

---

## Safety

- **Dry-run mode** is ON by default — no real orders until you explicitly enable auto-trading
- **Risk limits** cap each trade at 2% of capital with maximum 5 concurrent positions
- **Stop losses** are calculated using ATR for dynamic risk management
- All trades and signals are **logged to Supabase** for full audit trail

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. Algorithmic trading involves significant financial risk. Past performance does not guarantee future results. Always consult a registered financial advisor before making investment decisions.

---

## License

MIT
