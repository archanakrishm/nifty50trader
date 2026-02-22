"""
Strategy & Signal Engine
Combines candlestick patterns, technical indicators, and ML predictions
to generate actionable BUY/SELL/HOLD signals with confidence scores.
"""
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

import pandas as pd
import numpy as np

from config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD, STOP_LOSS_PCT,
    TAKE_PROFIT_PCT, TRAILING_STOP_PCT, RISK_PER_TRADE,
    MAX_OPEN_POSITIONS
)

logger = logging.getLogger(__name__)


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradeSignal:
    symbol: str
    signal: SignalType
    confidence: float          # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    timeframe: str = "daily"
    pattern_score: int = 0
    indicator_score: float = 0.0
    ml_prediction: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        return d


class StrategyEngine:
    """Multi-factor signal generator combining patterns, indicators, and ML."""

    def __init__(self, df: pd.DataFrame, symbol: str = "NIFTY50"):
        """
        Args:
            df: Enriched DataFrame with indicators + patterns already computed.
            symbol: Trading symbol.
        """
        self.df = df.copy()
        self.symbol = symbol

    # ─── Indicator Scoring ────────────────────────────────────────────────────
    def _score_rsi(self, row: pd.Series) -> tuple:
        """Score based on RSI."""
        rsi = row.get("rsi")
        if rsi is None or pd.isna(rsi):
            return 0.0, []
        if rsi < RSI_OVERSOLD:
            return 2.0, [f"RSI oversold ({rsi:.1f})"]
        elif rsi < 40:
            return 1.0, [f"RSI approaching oversold ({rsi:.1f})"]
        elif rsi > RSI_OVERBOUGHT:
            return -2.0, [f"RSI overbought ({rsi:.1f})"]
        elif rsi > 60:
            return -1.0, [f"RSI approaching overbought ({rsi:.1f})"]
        return 0.0, []

    def _score_macd(self, row: pd.Series) -> tuple:
        """Score based on MACD crossover."""
        macd = row.get("macd")
        sig = row.get("macd_signal")
        hist = row.get("macd_hist")
        if any(pd.isna(v) for v in [macd, sig, hist] if v is not None):
            return 0.0, []
        if macd is None:
            return 0.0, []

        score = 0.0
        reasons = []
        if macd > sig:
            score += 1.5
            reasons.append("MACD bullish crossover")
        else:
            score -= 1.5
            reasons.append("MACD bearish crossover")

        if hist is not None and not pd.isna(hist):
            if hist > 0:
                score += 0.5
                reasons.append("MACD histogram positive")
            else:
                score -= 0.5
                reasons.append("MACD histogram negative")
        return score, reasons

    def _score_moving_averages(self, row: pd.Series) -> tuple:
        """Score based on moving average alignment."""
        close = row.get("close")
        sma20 = row.get("sma_20")
        sma50 = row.get("sma_50")
        ema12 = row.get("ema_12")
        ema26 = row.get("ema_26")

        score = 0.0
        reasons = []

        if close and sma20 and not pd.isna(sma20):
            if close > sma20:
                score += 1.0
                reasons.append("Price above SMA20")
            else:
                score -= 1.0
                reasons.append("Price below SMA20")

        if sma20 and sma50 and not pd.isna(sma50):
            if sma20 > sma50:
                score += 1.5
                reasons.append("Golden cross (SMA20 > SMA50)")
            else:
                score -= 1.5
                reasons.append("Death cross (SMA20 < SMA50)")

        if ema12 and ema26 and not pd.isna(ema26):
            if ema12 > ema26:
                score += 1.0
                reasons.append("EMA12 > EMA26 (bullish)")
            else:
                score -= 1.0
                reasons.append("EMA12 < EMA26 (bearish)")

        return score, reasons

    def _score_bollinger(self, row: pd.Series) -> tuple:
        """Score based on Bollinger Band position."""
        close = row.get("close")
        bb_upper = row.get("bb_upper")
        bb_lower = row.get("bb_lower")
        bb_pct = row.get("bb_pct")

        if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in [bb_upper, bb_lower]):
            return 0.0, []

        score = 0.0
        reasons = []
        if close <= bb_lower:
            score += 2.0
            reasons.append("Price at lower Bollinger Band (oversold)")
        elif close >= bb_upper:
            score -= 2.0
            reasons.append("Price at upper Bollinger Band (overbought)")
        elif bb_pct is not None and not pd.isna(bb_pct):
            if bb_pct < 0.2:
                score += 1.0
                reasons.append("Price near lower BB")
            elif bb_pct > 0.8:
                score -= 1.0
                reasons.append("Price near upper BB")
        return score, reasons

    def _score_supertrend(self, row: pd.Series) -> tuple:
        """Score based on SuperTrend direction."""
        direction = row.get("supertrend_dir")
        if direction is None or (isinstance(direction, float) and pd.isna(direction)):
            return 0.0, []
        if direction == 1:
            return 2.0, ["SuperTrend bullish"]
        else:
            return -2.0, ["SuperTrend bearish"]

    def _score_adx(self, row: pd.Series) -> tuple:
        """Score based on ADX trend strength."""
        adx = row.get("adx")
        plus_di = row.get("plus_di")
        minus_di = row.get("minus_di")

        if adx is None or pd.isna(adx):
            return 0.0, []

        score = 0.0
        reasons = []
        if adx > 25:
            reasons.append(f"Strong trend (ADX={adx:.1f})")
            if plus_di and minus_di and not pd.isna(plus_di):
                if plus_di > minus_di:
                    score += 1.5
                    reasons.append("+DI > -DI (bullish trend)")
                else:
                    score -= 1.5
                    reasons.append("-DI > +DI (bearish trend)")
        else:
            reasons.append(f"Weak/no trend (ADX={adx:.1f})")
        return score, reasons

    def _score_stochastic(self, row: pd.Series) -> tuple:
        """Score based on Stochastic Oscillator."""
        k = row.get("stoch_k")
        d = row.get("stoch_d")
        if k is None or pd.isna(k):
            return 0.0, []
        score = 0.0
        reasons = []
        if k < 20:
            score += 1.5
            reasons.append(f"Stochastic oversold ({k:.1f})")
        elif k > 80:
            score -= 1.5
            reasons.append(f"Stochastic overbought ({k:.1f})")
        if k and d and not pd.isna(d):
            if k > d and k < 50:
                score += 0.5
                reasons.append("Stochastic bullish crossover")
            elif k < d and k > 50:
                score -= 0.5
                reasons.append("Stochastic bearish crossover")
        return score, reasons

    def _score_volume(self, row: pd.Series) -> tuple:
        """Score based on volume analysis."""
        vol = row.get("volume")
        vol_sma = row.get("volume_sma")
        if vol is None or vol_sma is None or pd.isna(vol_sma) or vol_sma == 0:
            return 0.0, []
        ratio = vol / vol_sma
        if ratio > 1.5:
            return 0.5, [f"High volume ({ratio:.1f}x avg) — confirms move"]
        return 0.0, []

    # ─── Composite Score → Signal ─────────────────────────────────────────────
    def generate_signal(self, idx: int = -1, ml_prediction: Optional[float] = None) -> TradeSignal:
        """Generate a trading signal for row at index `idx`."""
        row = self.df.iloc[idx]
        close = row["close"]

        total_score = 0.0
        all_reasons = []

        # Score every factor
        for scorer in [
            self._score_rsi,
            self._score_macd,
            self._score_moving_averages,
            self._score_bollinger,
            self._score_supertrend,
            self._score_adx,
            self._score_stochastic,
            self._score_volume,
        ]:
            s, r = scorer(row)
            total_score += s
            all_reasons.extend(r)

        # Candlestick pattern score
        pattern_score = row.get("pattern_score", 0)
        if not pd.isna(pattern_score):
            total_score += pattern_score
            patterns = row.get("patterns", [])
            if patterns:
                all_reasons.append(f"Candle patterns: {', '.join(patterns)}")

        # ML prediction boost
        if ml_prediction is not None:
            if ml_prediction > 0.6:
                total_score += 2.0
                all_reasons.append(f"ML predicts UP ({ml_prediction:.1%})")
            elif ml_prediction < 0.4:
                total_score -= 2.0
                all_reasons.append(f"ML predicts DOWN ({ml_prediction:.1%})")

        # Map score to signal
        max_possible = 18.0  # rough max sum
        confidence = min(abs(total_score) / max_possible, 1.0)

        if total_score >= 6:
            signal = SignalType.STRONG_BUY
        elif total_score >= 2:
            signal = SignalType.BUY
        elif total_score <= -6:
            signal = SignalType.STRONG_SELL
        elif total_score <= -2:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        # Risk management levels
        atr = row.get("atr", close * 0.02)
        if pd.isna(atr):
            atr = close * 0.02
        stop_loss = close - 2 * atr if signal in (SignalType.BUY, SignalType.STRONG_BUY) else close + 2 * atr
        take_profit = close + 3 * atr if signal in (SignalType.BUY, SignalType.STRONG_BUY) else close - 3 * atr

        return TradeSignal(
            symbol=self.symbol,
            signal=signal,
            confidence=round(confidence, 3),
            entry_price=round(close, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            reasons=all_reasons,
            pattern_score=int(pattern_score) if not pd.isna(pattern_score) else 0,
            indicator_score=round(total_score, 2),
            ml_prediction=ml_prediction,
        )

    def scan_all_signals(self, lookback: int = 5, ml_predictions: Optional[Dict] = None) -> List[TradeSignal]:
        """Generate signals for last `lookback` rows."""
        signals = []
        start = max(-lookback, -len(self.df))
        for i in range(start, 0):
            ml_pred = ml_predictions.get(i) if ml_predictions else None
            sig = self.generate_signal(idx=i, ml_prediction=ml_pred)
            signals.append(sig)
        return signals


class PortfolioManager:
    """Manage position sizing, risk, and trade execution decisions."""

    def __init__(self, capital: float = 100000.0):
        self.capital = capital
        self.positions: List[Dict] = []

    def calculate_position_size(self, signal: TradeSignal) -> Dict:
        """Calculate position size based on risk management rules."""
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return {"qty": 0, "reason": "Max positions reached"}

        risk_amount = self.capital * RISK_PER_TRADE
        price_risk = abs(signal.entry_price - signal.stop_loss)
        if price_risk == 0:
            return {"qty": 0, "reason": "Invalid stop loss"}

        qty = int(risk_amount / price_risk)
        total_cost = qty * signal.entry_price

        if total_cost > self.capital * 0.3:  # max 30% in single trade
            qty = int((self.capital * 0.3) / signal.entry_price)

        return {
            "qty": qty,
            "entry": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_amount": round(risk_amount, 2),
            "total_cost": round(qty * signal.entry_price, 2),
            "risk_reward_ratio": round(
                abs(signal.take_profit - signal.entry_price) / price_risk, 2
            ) if price_risk > 0 else 0,
        }

    def should_execute(self, signal: TradeSignal) -> bool:
        """Decide whether a signal is strong enough to trade."""
        if signal.signal in (SignalType.STRONG_BUY, SignalType.STRONG_SELL):
            return signal.confidence >= 0.4
        if signal.signal in (SignalType.BUY, SignalType.SELL):
            return signal.confidence >= 0.55
        return False
