"""
Claude AI Market Analyst
Uses Anthropic Claude to provide intelligent market analysis, predictions,
and investment strategy recommendations based on live data & indicators.
"""
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)


class ClaudeAnalyst:
    """Claude-powered market intelligence for Nifty 50 analysis."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "Anthropic API key not set. Add ANTHROPIC_API_KEY to your "
                ".env file (local) or Streamlit Secrets (cloud)."
            )
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL

    def _call(self, system: str, prompt: str, max_tokens: int = 4096) -> str:
        """Make a Claude API call."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error: {e}"

    # ─── Data Helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _df_summary(df: pd.DataFrame, last_n: int = 10) -> str:
        """Create a compact text summary of recent OHLCV + indicators."""
        recent = df.tail(last_n).copy()
        cols_to_show = [
            "date", "open", "high", "low", "close", "volume",
            "rsi", "macd", "macd_signal", "macd_hist",
            "sma_20", "sma_50", "ema_12", "ema_26",
            "bb_upper", "bb_lower", "atr", "adx",
            "supertrend_dir", "stoch_k", "stoch_d",
            "patterns", "pattern_score"
        ]
        available = [c for c in cols_to_show if c in recent.columns]
        subset = recent[available]

        lines = []
        for _, row in subset.iterrows():
            parts = []
            for col in available:
                val = row[col]
                if isinstance(val, (list, tuple)):
                    if val:
                        parts.append(f"{col}={','.join(str(v) for v in val)}")
                    continue
                try:
                    if pd.isna(val):
                        continue
                except (ValueError, TypeError):
                    pass
                if isinstance(val, float):
                    parts.append(f"{col}={val:.2f}")
                else:
                    parts.append(f"{col}={val}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _signal_summary(signal) -> str:
        """Summarize a TradeSignal as text."""
        return (
            f"Signal: {signal.signal.value}\n"
            f"Confidence: {signal.confidence:.1%}\n"
            f"Entry: ₹{signal.entry_price:,.2f}\n"
            f"Stop Loss: ₹{signal.stop_loss:,.2f}\n"
            f"Take Profit: ₹{signal.take_profit:,.2f}\n"
            f"Score: {signal.indicator_score:+.1f}\n"
            f"Pattern Score: {signal.pattern_score:+d}\n"
            f"Reasons:\n" + "\n".join(f"  - {r}" for r in signal.reasons)
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ANALYSIS METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def analyze_market(
        self,
        df: pd.DataFrame,
        signal,
        symbol: str = "NIFTY 50",
        quote: Optional[Dict] = None,
    ) -> str:
        """
        Comprehensive market analysis with prediction.

        Args:
            df: Enriched DataFrame with indicators + patterns.
            signal: TradeSignal from the strategy engine.
            symbol: Current symbol being analyzed.
            quote: Live quote dict (optional).

        Returns:
            Claude's detailed market analysis as markdown text.
        """
        system = (
            "You are an expert Indian stock market analyst specializing in Nifty 50 "
            "technical analysis. You analyze candlestick patterns, technical indicators, "
            "and market data to provide actionable insights. You give clear, specific "
            "predictions with reasoning. Always include:\n"
            "1. Current market assessment (bullish/bearish/neutral)\n"
            "2. Key indicator readings and what they mean\n"
            "3. Notable candlestick patterns and their implications\n"
            "4. Short-term prediction (1-5 days) with specific price targets\n"
            "5. Medium-term outlook (1-4 weeks)\n"
            "6. Recommended action (BUY/SELL/HOLD) with entry, SL, targets\n"
            "7. Risk factors to watch\n"
            "Format your response in clean markdown with headers."
        )

        data_summary = self._df_summary(df, last_n=15)
        signal_text = self._signal_summary(signal)

        quote_text = ""
        if quote:
            ltp = quote.get("last_price", "N/A")
            prev = quote.get("prev_close", "N/A")
            quote_text = (
                f"\nLive Quote:\n"
                f"  LTP: ₹{ltp}\n"
                f"  Prev Close: ₹{prev}\n"
                f"  Day High: ₹{quote.get('high', 'N/A')}\n"
                f"  Day Low: ₹{quote.get('low', 'N/A')}\n"
                f"  52W High: ₹{quote.get('fifty_two_week_high', 'N/A')}\n"
                f"  52W Low: ₹{quote.get('fifty_two_week_low', 'N/A')}\n"
            )

        prompt = (
            f"Analyze {symbol} as of {datetime.now().strftime('%d %B %Y')}.\n\n"
            f"=== RECENT OHLCV + INDICATORS (last 15 sessions) ===\n{data_summary}\n\n"
            f"=== STRATEGY ENGINE SIGNAL ===\n{signal_text}\n"
            f"{quote_text}\n"
            f"Provide a detailed technical analysis and market prediction."
        )

        return self._call(system, prompt)

    def predict_stock(
        self,
        df: pd.DataFrame,
        signal,
        symbol: str,
    ) -> str:
        """Quick stock-specific prediction."""
        system = (
            "You are an expert Indian equity analyst. Based on the technical data provided, "
            "give a concise prediction in this exact format:\n"
            "VERDICT: [STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL]\n"
            "CONFIDENCE: [percentage]\n"
            "TARGET (1 week): ₹[price]\n"
            "TARGET (1 month): ₹[price]\n"
            "STOP LOSS: ₹[price]\n"
            "KEY REASON: [1-2 sentence summary]\n\n"
            "Then provide a brief 3-4 line explanation."
        )

        data_summary = self._df_summary(df, last_n=10)
        signal_text = self._signal_summary(signal)

        prompt = (
            f"Stock: {symbol}\n"
            f"Date: {datetime.now().strftime('%d %B %Y')}\n\n"
            f"Recent data:\n{data_summary}\n\n"
            f"Strategy signal:\n{signal_text}\n\n"
            f"Give your prediction."
        )

        return self._call(system, prompt, max_tokens=1024)

    def investment_strategy(
        self,
        nifty_df: pd.DataFrame,
        nifty_signal,
        stock_signals: Optional[List[Dict]] = None,
        portfolio: Optional[Dict] = None,
    ) -> str:
        """
        Generate a comprehensive investment strategy.

        Args:
            nifty_df: Nifty 50 index enriched DataFrame.
            nifty_signal: Nifty 50 TradeSignal.
            stock_signals: List of {symbol, signal_value, confidence, price} dicts.
            portfolio: Current portfolio info (capital, positions).
        """
        system = (
            "You are a SEBI-registered investment advisor (simulated) specializing in "
            "Nifty 50 stocks. Based on the market data and signals provided, create a "
            "detailed investment strategy. Include:\n\n"
            "1. **Market Regime**: Current market condition (trending/ranging, bull/bear)\n"
            "2. **Sector Rotation**: Which sectors to prefer/avoid\n"
            "3. **Top Picks**: 3-5 specific stock recommendations with entry/SL/target\n"
            "4. **Portfolio Allocation**: How to split capital across picks\n"
            "5. **Risk Management**: Position sizing, hedging suggestions\n"
            "6. **Timeline**: Expected holding period\n"
            "7. **Exit Plan**: When and why to exit\n\n"
            "Be specific with numbers. Use Indian market conventions (NSE, ₹)."
        )

        data_summary = self._df_summary(nifty_df, last_n=10)
        signal_text = self._signal_summary(nifty_signal)

        stocks_text = ""
        if stock_signals:
            stocks_text = "\n=== STOCK SCANNER RESULTS ===\n"
            for s in stock_signals[:15]:
                stocks_text += (
                    f"  {s.get('symbol', '?')}: {s.get('signal', '?')} "
                    f"(conf={s.get('confidence', '?')}, price=₹{s.get('price', '?')}, "
                    f"score={s.get('score', '?')})\n"
                )

        portfolio_text = ""
        if portfolio:
            portfolio_text = (
                f"\n=== CURRENT PORTFOLIO ===\n"
                f"  Capital: ₹{portfolio.get('capital', 100000):,}\n"
                f"  Invested: ₹{portfolio.get('invested', 0):,}\n"
                f"  Cash: ₹{portfolio.get('cash', 100000):,}\n"
                f"  Open positions: {portfolio.get('positions', 0)}\n"
            )

        prompt = (
            f"Create an investment strategy for {datetime.now().strftime('%d %B %Y')}.\n\n"
            f"=== NIFTY 50 DATA (last 10 sessions) ===\n{data_summary}\n\n"
            f"=== NIFTY 50 SIGNAL ===\n{signal_text}\n"
            f"{stocks_text}"
            f"{portfolio_text}\n"
            f"Generate a complete investment strategy."
        )

        return self._call(system, prompt)

    def explain_signal(self, signal) -> str:
        """Get Claude to explain a signal in simple terms."""
        system = (
            "You are a friendly stock market educator. Explain the trading signal "
            "in simple terms that a beginner investor can understand. "
            "Use analogies, avoid jargon, and be concise (5-8 sentences)."
        )
        signal_text = self._signal_summary(signal)
        prompt = f"Explain this trading signal to a beginner:\n\n{signal_text}"
        return self._call(system, prompt, max_tokens=512)

    def ask_market_question(self, question: str, df: pd.DataFrame, signal) -> str:
        """Answer any market-related question with context."""
        system = (
            "You are an expert Indian stock market analyst with access to live "
            "Nifty 50 data. Answer the user's question using the provided market "
            "data and indicators. Be specific and data-driven."
        )
        data_summary = self._df_summary(df, last_n=10)
        signal_text = self._signal_summary(signal)
        prompt = (
            f"Market context:\n{data_summary}\n\n"
            f"Current signal:\n{signal_text}\n\n"
            f"User's question: {question}"
        )
        return self._call(system, prompt)

    def nifty50_scanner_analysis(self, scanner_results: List[Dict]) -> str:
        """Analyze batch scanner results and pick the best opportunities."""
        system = (
            "You are a stock screener analyst. Given the scan results of Nifty 50 stocks "
            "with their technical signals, identify:\n"
            "1. Top 3 BUY opportunities with reasoning\n"
            "2. Top 3 stocks to AVOID/SELL\n"
            "3. Overall market breadth assessment\n"
            "4. Sector trends visible from the data\n"
            "Be specific with prices and targets."
        )
        lines = []
        for s in scanner_results:
            lines.append(
                f"{s.get('Symbol', '?')}: {s.get('Signal', '?')} | "
                f"Conf={s.get('Confidence', '?')} | Score={s.get('Score', '?')} | "
                f"Price={s.get('Price', '?')} | SL={s.get('SL', '?')} | TP={s.get('TP', '?')}"
            )
        prompt = f"Analyze these Nifty 50 stock scanner results:\n\n" + "\n".join(lines)
        return self._call(system, prompt)
