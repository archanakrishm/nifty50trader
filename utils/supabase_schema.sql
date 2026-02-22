-- =====================================================================
-- Supabase Schema for Nifty50 Trader
-- Run this in the Supabase SQL Editor to create all required tables
-- =====================================================================

-- 1. Trade Signals — every generated signal gets stored
CREATE TABLE IF NOT EXISTS trade_signals (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    signal          TEXT NOT NULL CHECK (signal IN ('STRONG_BUY','BUY','HOLD','SELL','STRONG_SELL')),
    confidence      FLOAT NOT NULL,
    entry_price     FLOAT NOT NULL,
    stop_loss       FLOAT,
    take_profit     FLOAT,
    reasons         JSONB DEFAULT '[]',
    pattern_score   INT DEFAULT 0,
    indicator_score FLOAT DEFAULT 0,
    ml_prediction   FLOAT,
    timeframe       TEXT DEFAULT 'daily',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_signals_symbol ON trade_signals(symbol);
CREATE INDEX idx_signals_created ON trade_signals(created_at DESC);

-- 2. Executed Trades — trades actually placed via broker
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       BIGINT REFERENCES trade_signals(id),
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
    qty             INT NOT NULL,
    entry_price     FLOAT NOT NULL,
    exit_price      FLOAT,
    stop_loss       FLOAT,
    take_profit     FLOAT,
    broker          TEXT NOT NULL,        -- 'zerodha', 'angelone', 'groww'
    broker_order_id TEXT,
    status          TEXT DEFAULT 'OPEN' CHECK (status IN ('OPEN','CLOSED','CANCELLED','FAILED')),
    pnl             FLOAT,
    pnl_pct         FLOAT,
    opened_at       TIMESTAMPTZ DEFAULT now(),
    closed_at       TIMESTAMPTZ,
    notes           TEXT
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);

-- 3. Portfolio Snapshot — periodic capital & holdings snapshot
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    total_capital   FLOAT NOT NULL,
    invested        FLOAT DEFAULT 0,
    available_cash  FLOAT DEFAULT 0,
    unrealized_pnl  FLOAT DEFAULT 0,
    realized_pnl    FLOAT DEFAULT 0,
    holdings        JSONB DEFAULT '{}',
    snapshot_at     TIMESTAMPTZ DEFAULT now()
);

-- 4. Market Data Cache — optional cache for historical candles
CREATE TABLE IF NOT EXISTS market_data_cache (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    date            TIMESTAMPTZ NOT NULL,
    open            FLOAT,
    high            FLOAT,
    low             FLOAT,
    close           FLOAT,
    volume          BIGINT,
    fetched_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(symbol, timeframe, date)
);

CREATE INDEX idx_market_cache_sym ON market_data_cache(symbol, timeframe, date);

-- 5. Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL UNIQUE,
    added_at        TIMESTAMPTZ DEFAULT now(),
    notes           TEXT
);

-- 6. Strategy Performance — daily strategy log
CREATE TABLE IF NOT EXISTS strategy_performance (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    total_signals   INT DEFAULT 0,
    buy_signals     INT DEFAULT 0,
    sell_signals    INT DEFAULT 0,
    trades_taken    INT DEFAULT 0,
    win_count       INT DEFAULT 0,
    loss_count      INT DEFAULT 0,
    total_pnl       FLOAT DEFAULT 0,
    win_rate        FLOAT DEFAULT 0,
    avg_return      FLOAT DEFAULT 0,
    max_drawdown    FLOAT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_perf_date ON strategy_performance(date DESC);

-- 7. App Settings — key-value user settings
CREATE TABLE IF NOT EXISTS app_settings (
    key             TEXT PRIMARY KEY,
    value           JSONB NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Insert default settings
INSERT INTO app_settings (key, value) VALUES
    ('auto_trade_enabled', 'false'::jsonb),
    ('risk_per_trade', '0.02'::jsonb),
    ('max_positions', '5'::jsonb),
    ('default_broker', '"zerodha"'::jsonb),
    ('capital', '100000'::jsonb)
ON CONFLICT (key) DO NOTHING;

-- 8. Enable Row Level Security (optional — uncomment for auth)
-- ALTER TABLE trade_signals ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
