"""
Technical Indicators Module
Computes RSI, MACD, Bollinger Bands, SuperTrend, ADX, VWAP, and more.
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    SMA_SHORT, SMA_LONG, EMA_SHORT, EMA_LONG,
    BOLLINGER_PERIOD, BOLLINGER_STD, ATR_PERIOD,
    STOCH_PERIOD, STOCH_SMOOTH, ADX_PERIOD,
    SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Compute all technical indicators on OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ─── Moving Averages ──────────────────────────────────────────────────────
    def add_sma(self, period: int = SMA_SHORT, col: str = "close") -> pd.DataFrame:
        self.df[f"sma_{period}"] = self.df[col].rolling(window=period).mean()
        return self.df

    def add_ema(self, period: int = EMA_SHORT, col: str = "close") -> pd.DataFrame:
        self.df[f"ema_{period}"] = self.df[col].ewm(span=period, adjust=False).mean()
        return self.df

    # ─── RSI ──────────────────────────────────────────────────────────────────
    def add_rsi(self, period: int = RSI_PERIOD, col: str = "close") -> pd.DataFrame:
        delta = self.df[col].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        self.df["rsi"] = 100 - (100 / (1 + rs))
        return self.df

    # ─── MACD ─────────────────────────────────────────────────────────────────
    def add_macd(
        self,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL,
        col: str = "close"
    ) -> pd.DataFrame:
        ema_fast = self.df[col].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df[col].ewm(span=slow, adjust=False).mean()
        self.df["macd"] = ema_fast - ema_slow
        self.df["macd_signal"] = self.df["macd"].ewm(span=signal, adjust=False).mean()
        self.df["macd_hist"] = self.df["macd"] - self.df["macd_signal"]
        return self.df

    # ─── Bollinger Bands ──────────────────────────────────────────────────────
    def add_bollinger_bands(
        self,
        period: int = BOLLINGER_PERIOD,
        std_dev: float = BOLLINGER_STD,
        col: str = "close"
    ) -> pd.DataFrame:
        sma = self.df[col].rolling(window=period).mean()
        std = self.df[col].rolling(window=period).std()
        self.df["bb_mid"] = sma
        self.df["bb_upper"] = sma + std_dev * std
        self.df["bb_lower"] = sma - std_dev * std
        self.df["bb_width"] = (self.df["bb_upper"] - self.df["bb_lower"]) / self.df["bb_mid"]
        self.df["bb_pct"] = (self.df[col] - self.df["bb_lower"]) / (self.df["bb_upper"] - self.df["bb_lower"])
        return self.df

    # ─── ATR (Average True Range) ─────────────────────────────────────────────
    def add_atr(self, period: int = ATR_PERIOD) -> pd.DataFrame:
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"].shift(1)
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df["atr"] = tr.rolling(window=period).mean()
        return self.df

    # ─── Stochastic Oscillator ────────────────────────────────────────────────
    def add_stochastic(
        self,
        period: int = STOCH_PERIOD,
        smooth: int = STOCH_SMOOTH
    ) -> pd.DataFrame:
        low_min = self.df["low"].rolling(window=period).min()
        high_max = self.df["high"].rolling(window=period).max()
        self.df["stoch_k"] = 100 * (self.df["close"] - low_min) / (high_max - low_min)
        self.df["stoch_d"] = self.df["stoch_k"].rolling(window=smooth).mean()
        return self.df

    # ─── ADX (Average Directional Index) ──────────────────────────────────────
    def add_adx(self, period: int = ADX_PERIOD) -> pd.DataFrame:
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # ATR for ADX
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        self.df["adx"] = dx.ewm(alpha=1 / period, min_periods=period).mean()
        self.df["plus_di"] = plus_di
        self.df["minus_di"] = minus_di
        return self.df

    # ─── OBV (On Balance Volume) ──────────────────────────────────────────────
    def add_obv(self) -> pd.DataFrame:
        direction = np.sign(self.df["close"].diff())
        self.df["obv"] = (direction * self.df["volume"]).cumsum()
        return self.df

    # ─── Volume SMA ───────────────────────────────────────────────────────────
    def add_volume_sma(self, period: int = 20) -> pd.DataFrame:
        self.df["volume_sma"] = self.df["volume"].rolling(window=period).mean()
        return self.df

    # ─── VWAP ─────────────────────────────────────────────────────────────────
    def add_vwap(self) -> pd.DataFrame:
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        cumulative_tp_vol = (typical_price * self.df["volume"]).cumsum()
        cumulative_vol = self.df["volume"].cumsum()
        self.df["vwap"] = cumulative_tp_vol / cumulative_vol
        return self.df

    # ─── SuperTrend ───────────────────────────────────────────────────────────
    def add_supertrend(
        self,
        period: int = SUPERTREND_PERIOD,
        multiplier: float = SUPERTREND_MULTIPLIER
    ) -> pd.DataFrame:
        hl2 = (self.df["high"] + self.df["low"]) / 2
        if "atr" not in self.df.columns:
            self.add_atr(period)
        atr = self.df["atr"]

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(index=self.df.index, dtype=int)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(self.df)):
            if self.df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif self.df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i],
                                         supertrend.iloc[i - 1] if direction.iloc[i - 1] == 1 else lower_band.iloc[i])
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i],
                                         supertrend.iloc[i - 1] if direction.iloc[i - 1] == -1 else upper_band.iloc[i])

        self.df["supertrend"] = supertrend
        self.df["supertrend_dir"] = direction
        return self.df

    # ─── Ichimoku Cloud ───────────────────────────────────────────────────────
    def add_ichimoku(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52
    ) -> pd.DataFrame:
        high = self.df["high"]
        low = self.df["low"]

        self.df["tenkan_sen"] = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        self.df["kijun_sen"] = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        self.df["senkou_a"] = ((self.df["tenkan_sen"] + self.df["kijun_sen"]) / 2).shift(kijun)
        self.df["senkou_b"] = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        self.df["chikou_span"] = self.df["close"].shift(-kijun)
        return self.df

    # ─── Fibonacci Retracement Levels ─────────────────────────────────────────
    def fibonacci_levels(self, lookback: int = 50) -> dict:
        """Calculate Fibonacci retracement levels from recent swing."""
        recent = self.df.tail(lookback)
        high = recent["high"].max()
        low = recent["low"].min()
        diff = high - low
        levels = {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "78.6%": high - 0.786 * diff,
            "100%": low,
        }
        return levels

    # ─── Support & Resistance ─────────────────────────────────────────────────
    def support_resistance(self, lookback: int = 50, num_levels: int = 3) -> dict:
        """Find support and resistance levels using pivot points."""
        recent = self.df.tail(lookback)
        pivots = (recent["high"] + recent["low"] + recent["close"]) / 3

        # Find local minima/maxima
        highs = recent["high"].values
        lows = recent["low"].values

        resistances = sorted(set(highs), reverse=True)[:num_levels]
        supports = sorted(set(lows))[:num_levels]

        return {
            "supports": supports,
            "resistances": resistances,
            "pivot": pivots.iloc[-1] if len(pivots) > 0 else None,
        }

    # ─── Add All Indicators ──────────────────────────────────────────────────
    def add_all(self) -> pd.DataFrame:
        """Compute every indicator and return enriched DataFrame."""
        self.add_sma(SMA_SHORT)
        self.add_sma(SMA_LONG)
        self.add_ema(EMA_SHORT)
        self.add_ema(EMA_LONG)
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_stochastic()
        self.add_adx()
        self.add_obv()
        self.add_volume_sma()
        self.add_vwap()
        self.add_supertrend()
        self.add_ichimoku()
        logger.info(f"All {len(self.df.columns)} indicators computed")
        return self.df
