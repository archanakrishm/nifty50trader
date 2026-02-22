"""
Data Fetcher Module
Fetches live and historical Nifty50 market data using yfinance and NSE APIs.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import yfinance as yf
import requests

from config import (
    NIFTY50_SYMBOL, NIFTY50_STOCKS, DEFAULT_PERIOD,
    DEFAULT_INTERVAL, INTRADAY_INTERVAL, INDEX_CONSTITUENTS_URL
)

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles all market data retrieval operations."""

    NSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.NSE_HEADERS)
        self._nse_cookies_set = False

    # ─── NSE Cookie Handshake ─────────────────────────────────────────────────
    def _init_nse_session(self):
        """Hit NSE homepage to get cookies before API calls."""
        if not self._nse_cookies_set:
            try:
                self._session.get("https://www.nseindia.com", timeout=10)
                self._nse_cookies_set = True
            except Exception as e:
                logger.warning(f"NSE session init failed: {e}")

    # ─── Nifty 50 Index Data ─────────────────────────────────────────────────
    def get_nifty50_history(
        self,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for Nifty 50 index.

        Args:
            period: Data lookback period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume columns
        """
        try:
            ticker = yf.Ticker(NIFTY50_SYMBOL)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning("No data returned for Nifty 50 index")
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
            logger.info(f"Fetched {len(df)} candles for Nifty 50 ({interval})")
            return df
        except Exception as e:
            logger.error(f"Error fetching Nifty 50 data: {e}")
            return pd.DataFrame()

    # ─── Individual Stock Data ────────────────────────────────────────────────
    def get_stock_history(
        self,
        symbol: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a single stock."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # ─── Batch Fetch All Nifty 50 Stocks ──────────────────────────────────────
    def get_all_nifty50_stocks(
        self,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        stocks: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all Nifty 50 constituent stocks."""
        stock_list = stocks or NIFTY50_STOCKS
        results = {}
        for symbol in stock_list:
            df = self.get_stock_history(symbol, period, interval)
            if not df.empty:
                results[symbol] = df
        logger.info(f"Fetched data for {len(results)}/{len(stock_list)} stocks")
        return results

    # ─── Live / Realtime Quote ────────────────────────────────────────────────
    def get_live_quote(self, symbol: str = NIFTY50_SYMBOL) -> Dict:
        """Get the latest live quote for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "last_price": getattr(info, "last_price", None),
                "open": getattr(info, "open", None),
                "high": getattr(info, "day_high", None),
                "low": getattr(info, "day_low", None),
                "prev_close": getattr(info, "previous_close", None),
                "volume": getattr(info, "last_volume", None),
                "market_cap": getattr(info, "market_cap", None),
                "fifty_two_week_high": getattr(info, "year_high", None),
                "fifty_two_week_low": getattr(info, "year_low", None),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting live quote for {symbol}: {e}")
            return {}

    # ─── Live Nifty 50 from NSE API ──────────────────────────────────────────
    def get_nse_live_nifty50(self) -> Dict:
        """Fetch live Nifty 50 data directly from NSE India API."""
        self._init_nse_session()
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            index_data = data.get("metadata", {})
            constituents = data.get("data", [])
            return {
                "index": {
                    "last": index_data.get("last"),
                    "open": index_data.get("open"),
                    "high": index_data.get("high"),
                    "low": index_data.get("low"),
                    "prev_close": index_data.get("previousClose"),
                    "change": index_data.get("change"),
                    "pchange": index_data.get("pChange"),
                    "timestamp": index_data.get("timeVal"),
                },
                "constituents": [
                    {
                        "symbol": stock.get("symbol"),
                        "last": stock.get("lastPrice"),
                        "open": stock.get("open"),
                        "high": stock.get("dayHigh"),
                        "low": stock.get("dayLow"),
                        "prev_close": stock.get("previousClose"),
                        "change": stock.get("change"),
                        "pchange": stock.get("pChange"),
                        "volume": stock.get("totalTradedVolume"),
                    }
                    for stock in constituents
                    if stock.get("symbol") != "NIFTY 50"
                ],
            }
        except Exception as e:
            logger.error(f"NSE live data error: {e}")
            return {}

    # ─── Nifty 50 Constituents List ──────────────────────────────────────────
    def get_nifty50_constituents(self) -> pd.DataFrame:
        """Fetch the official Nifty 50 constituents list from NSE."""
        try:
            df = pd.read_csv(INDEX_CONSTITUENTS_URL)
            return df
        except Exception as e:
            logger.error(f"Error fetching constituents: {e}")
            return pd.DataFrame()

    # ─── Intraday Data ────────────────────────────────────────────────────────
    def get_intraday_data(
        self,
        symbol: str = NIFTY50_SYMBOL,
        interval: str = INTRADAY_INTERVAL,
        days: int = 5
    ) -> pd.DataFrame:
        """Fetch intraday OHLCV data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval=interval)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
            return pd.DataFrame()

    # ─── Multi-timeframe Data ─────────────────────────────────────────────────
    def get_multi_timeframe(
        self,
        symbol: str = NIFTY50_SYMBOL
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data across multiple timeframes for comprehensive analysis."""
        timeframes = {
            "5min": ("5d", "5m"),
            "15min": ("5d", "15m"),
            "1hour": ("1mo", "1h"),
            "daily": ("6mo", "1d"),
            "weekly": ("2y", "1wk"),
        }
        results = {}
        for tf_name, (period, interval) in timeframes.items():
            df = self.get_nifty50_history(period=period, interval=interval) \
                if symbol == NIFTY50_SYMBOL \
                else self.get_stock_history(symbol, period, interval)
            if not df.empty:
                results[tf_name] = df
        return results

    # ─── Options Chain Data (NSE) ─────────────────────────────────────────────
    def get_options_chain(self, symbol: str = "NIFTY") -> Dict:
        """Fetch options chain data from NSE."""
        self._init_nse_session()
        try:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", {})
            return {
                "expiry_dates": records.get("expiryDates", []),
                "strikePrices": records.get("strikePrices", []),
                "underlying_value": records.get("underlyingValue"),
                "data": records.get("data", []),
                "timestamp": records.get("timestamp"),
            }
        except Exception as e:
            logger.error(f"Options chain error: {e}")
            return {}
