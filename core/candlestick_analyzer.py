"""
Candlestick Pattern Analyzer
Identifies and classifies Japanese candlestick patterns for trading signals.
"""
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CandlestickAnalyzer:
    """Detects candlestick patterns and assigns bullish/bearish scores."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self._ensure_columns()

    def _ensure_columns(self):
        required = {"open", "high", "low", "close"}
        cols = set(self.df.columns)
        if not required.issubset(cols):
            raise ValueError(f"Missing columns: {required - cols}")

    # ─── Helper Properties ────────────────────────────────────────────────────
    def _body(self, i: int) -> float:
        return abs(self.df["close"].iloc[i] - self.df["open"].iloc[i])

    def _upper_shadow(self, i: int) -> float:
        return self.df["high"].iloc[i] - max(self.df["close"].iloc[i], self.df["open"].iloc[i])

    def _lower_shadow(self, i: int) -> float:
        return min(self.df["close"].iloc[i], self.df["open"].iloc[i]) - self.df["low"].iloc[i]

    def _is_bullish(self, i: int) -> bool:
        return self.df["close"].iloc[i] > self.df["open"].iloc[i]

    def _is_bearish(self, i: int) -> bool:
        return self.df["close"].iloc[i] < self.df["open"].iloc[i]

    def _candle_range(self, i: int) -> float:
        return self.df["high"].iloc[i] - self.df["low"].iloc[i]

    def _avg_body(self, i: int, lookback: int = 14) -> float:
        start = max(0, i - lookback)
        bodies = [self._body(j) for j in range(start, i)]
        return np.mean(bodies) if bodies else self._body(i)

    # ─── Single Candle Patterns ───────────────────────────────────────────────
    def detect_doji(self, i: int) -> Optional[str]:
        """Doji: body is very small relative to range."""
        body = self._body(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        if body / rng < 0.1:
            return "doji"
        return None

    def detect_hammer(self, i: int) -> Optional[str]:
        """Hammer: small body at top, long lower shadow (bullish reversal)."""
        body = self._body(i)
        lower = self._lower_shadow(i)
        upper = self._upper_shadow(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        if lower >= 2 * body and upper <= body * 0.3 and body / rng < 0.35:
            return "hammer"
        return None

    def detect_inverted_hammer(self, i: int) -> Optional[str]:
        """Inverted Hammer: small body at bottom, long upper shadow."""
        body = self._body(i)
        upper = self._upper_shadow(i)
        lower = self._lower_shadow(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        if upper >= 2 * body and lower <= body * 0.3 and body / rng < 0.35:
            return "inverted_hammer"
        return None

    def detect_shooting_star(self, i: int) -> Optional[str]:
        """Shooting star: long upper shadow, small body at bottom (bearish)."""
        if i < 1:
            return None
        body = self._body(i)
        upper = self._upper_shadow(i)
        lower = self._lower_shadow(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        prev_bullish = self._is_bullish(i - 1)
        if upper >= 2 * body and lower <= body * 0.3 and prev_bullish:
            return "shooting_star"
        return None

    def detect_hanging_man(self, i: int) -> Optional[str]:
        """Hanging man: hammer shape in an uptrend (bearish)."""
        if i < 1:
            return None
        body = self._body(i)
        lower = self._lower_shadow(i)
        upper = self._upper_shadow(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        prev_bullish = self._is_bullish(i - 1)
        if lower >= 2 * body and upper <= body * 0.3 and prev_bullish:
            return "hanging_man"
        return None

    def detect_marubozu(self, i: int) -> Optional[str]:
        """Marubozu: large body with no/tiny shadows."""
        body = self._body(i)
        upper = self._upper_shadow(i)
        lower = self._lower_shadow(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        if body / rng > 0.9:
            return "bullish_marubozu" if self._is_bullish(i) else "bearish_marubozu"
        return None

    def detect_spinning_top(self, i: int) -> Optional[str]:
        """Spinning top: small body, shadows on both sides."""
        body = self._body(i)
        upper = self._upper_shadow(i)
        lower = self._lower_shadow(i)
        rng = self._candle_range(i)
        if rng == 0:
            return None
        if body / rng < 0.3 and upper > body * 0.5 and lower > body * 0.5:
            return "spinning_top"
        return None

    # ─── Two-Candle Patterns ──────────────────────────────────────────────────
    def detect_engulfing(self, i: int) -> Optional[str]:
        """Bullish/Bearish engulfing patterns."""
        if i < 1:
            return None
        curr_body = self._body(i)
        prev_body = self._body(i - 1)
        if curr_body <= prev_body:
            return None
        c_open, c_close = self.df["open"].iloc[i], self.df["close"].iloc[i]
        p_open, p_close = self.df["open"].iloc[i - 1], self.df["close"].iloc[i - 1]

        if self._is_bullish(i) and self._is_bearish(i - 1):
            if c_open <= p_close and c_close >= p_open:
                return "bullish_engulfing"
        elif self._is_bearish(i) and self._is_bullish(i - 1):
            if c_open >= p_close and c_close <= p_open:
                return "bearish_engulfing"
        return None

    def detect_harami(self, i: int) -> Optional[str]:
        """Bullish/Bearish harami patterns."""
        if i < 1:
            return None
        curr_body = self._body(i)
        prev_body = self._body(i - 1)
        if curr_body >= prev_body:
            return None
        c_open, c_close = self.df["open"].iloc[i], self.df["close"].iloc[i]
        p_open, p_close = self.df["open"].iloc[i - 1], self.df["close"].iloc[i - 1]

        if self._is_bullish(i) and self._is_bearish(i - 1):
            if min(c_open, c_close) >= min(p_open, p_close) and max(c_open, c_close) <= max(p_open, p_close):
                return "bullish_harami"
        elif self._is_bearish(i) and self._is_bullish(i - 1):
            if min(c_open, c_close) >= min(p_open, p_close) and max(c_open, c_close) <= max(p_open, p_close):
                return "bearish_harami"
        return None

    def detect_tweezer(self, i: int) -> Optional[str]:
        """Tweezer top/bottom patterns."""
        if i < 1:
            return None
        tolerance = self._candle_range(i) * 0.02
        if abs(self.df["low"].iloc[i] - self.df["low"].iloc[i - 1]) <= tolerance:
            if self._is_bearish(i - 1) and self._is_bullish(i):
                return "tweezer_bottom"
        if abs(self.df["high"].iloc[i] - self.df["high"].iloc[i - 1]) <= tolerance:
            if self._is_bullish(i - 1) and self._is_bearish(i):
                return "tweezer_top"
        return None

    def detect_piercing_dark_cloud(self, i: int) -> Optional[str]:
        """Piercing line (bullish) / Dark cloud cover (bearish)."""
        if i < 1:
            return None
        p_open, p_close = self.df["open"].iloc[i - 1], self.df["close"].iloc[i - 1]
        c_open, c_close = self.df["open"].iloc[i], self.df["close"].iloc[i]
        p_mid = (p_open + p_close) / 2

        if self._is_bearish(i - 1) and self._is_bullish(i):
            if c_open < p_close and c_close > p_mid and c_close < p_open:
                return "piercing_line"
        if self._is_bullish(i - 1) and self._is_bearish(i):
            if c_open > p_close and c_close < p_mid and c_close > p_open:
                return "dark_cloud_cover"
        return None

    # ─── Three-Candle Patterns ────────────────────────────────────────────────
    def detect_morning_evening_star(self, i: int) -> Optional[str]:
        """Morning star (bullish) / Evening star (bearish)."""
        if i < 2:
            return None
        first_body = self._body(i - 2)
        second_body = self._body(i - 1)
        third_body = self._body(i)
        avg = self._avg_body(i)

        # Morning star
        if (self._is_bearish(i - 2) and first_body > avg * 0.8 and
                second_body < avg * 0.4 and
                self._is_bullish(i) and third_body > avg * 0.8):
            if self.df["close"].iloc[i] > (self.df["open"].iloc[i - 2] + self.df["close"].iloc[i - 2]) / 2:
                return "morning_star"

        # Evening star
        if (self._is_bullish(i - 2) and first_body > avg * 0.8 and
                second_body < avg * 0.4 and
                self._is_bearish(i) and third_body > avg * 0.8):
            if self.df["close"].iloc[i] < (self.df["open"].iloc[i - 2] + self.df["close"].iloc[i - 2]) / 2:
                return "evening_star"
        return None

    def detect_three_soldiers_crows(self, i: int) -> Optional[str]:
        """Three white soldiers (bullish) / Three black crows (bearish)."""
        if i < 2:
            return None
        # Three white soldiers
        if all(self._is_bullish(i - j) for j in range(3)):
            if (self.df["close"].iloc[i] > self.df["close"].iloc[i - 1] >
                    self.df["close"].iloc[i - 2]):
                bodies = [self._body(i - j) for j in range(3)]
                avg = self._avg_body(i)
                if all(b > avg * 0.5 for b in bodies):
                    return "three_white_soldiers"

        # Three black crows
        if all(self._is_bearish(i - j) for j in range(3)):
            if (self.df["close"].iloc[i] < self.df["close"].iloc[i - 1] <
                    self.df["close"].iloc[i - 2]):
                bodies = [self._body(i - j) for j in range(3)]
                avg = self._avg_body(i)
                if all(b > avg * 0.5 for b in bodies):
                    return "three_black_crows"
        return None

    # ─── Master Analyzer ──────────────────────────────────────────────────────
    def analyze_all(self) -> pd.DataFrame:
        """Run all pattern detectors across the entire DataFrame.

        Returns a DataFrame with a 'patterns' column (list of pattern names)
        and a 'pattern_score' column (net score: bullish positive, bearish negative).
        """
        pattern_funcs = [
            self.detect_doji,
            self.detect_hammer,
            self.detect_inverted_hammer,
            self.detect_shooting_star,
            self.detect_hanging_man,
            self.detect_marubozu,
            self.detect_spinning_top,
            self.detect_engulfing,
            self.detect_harami,
            self.detect_tweezer,
            self.detect_piercing_dark_cloud,
            self.detect_morning_evening_star,
            self.detect_three_soldiers_crows,
        ]

        BULLISH_PATTERNS = {
            "hammer", "inverted_hammer", "bullish_engulfing", "bullish_harami",
            "tweezer_bottom", "piercing_line", "morning_star",
            "three_white_soldiers", "bullish_marubozu", "doji"
        }
        BEARISH_PATTERNS = {
            "shooting_star", "hanging_man", "bearish_engulfing", "bearish_harami",
            "tweezer_top", "dark_cloud_cover", "evening_star",
            "three_black_crows", "bearish_marubozu"
        }
        SCORE_MAP = {
            # Bullish
            "hammer": 2, "inverted_hammer": 1, "bullish_engulfing": 3,
            "bullish_harami": 1, "tweezer_bottom": 2, "piercing_line": 2,
            "morning_star": 3, "three_white_soldiers": 3, "bullish_marubozu": 2,
            "doji": 0,
            # Bearish
            "shooting_star": -2, "hanging_man": -2, "bearish_engulfing": -3,
            "bearish_harami": -1, "tweezer_top": -2, "dark_cloud_cover": -2,
            "evening_star": -3, "three_black_crows": -3, "bearish_marubozu": -2,
            "spinning_top": 0,
        }

        all_patterns = []
        all_scores = []

        for idx in range(len(self.df)):
            found = []
            for func in pattern_funcs:
                result = func(idx)
                if result:
                    found.append(result)
            all_patterns.append(found)
            score = sum(SCORE_MAP.get(p, 0) for p in found)
            all_scores.append(score)

        self.df["patterns"] = all_patterns
        self.df["pattern_score"] = all_scores
        logger.info(f"Analyzed {len(self.df)} candles for patterns")
        return self.df

    def get_latest_signals(self, n: int = 5) -> List[Dict]:
        """Get the most recent candle pattern signals."""
        if "patterns" not in self.df.columns:
            self.analyze_all()
        recent = self.df.tail(n)
        signals = []
        for _, row in recent.iterrows():
            if row["patterns"]:
                signals.append({
                    "date": str(row.get("date", "")),
                    "patterns": row["patterns"],
                    "score": row["pattern_score"],
                    "close": row["close"],
                    "signal": "BULLISH" if row["pattern_score"] > 0 else
                              "BEARISH" if row["pattern_score"] < 0 else "NEUTRAL"
                })
        return signals
