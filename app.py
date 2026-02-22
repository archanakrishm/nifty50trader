"""
Nifty50 Trader — Streamlit Dashboard
Main UI with live data, candlestick charts, signals, ML predictions,
broker integration, and trade management — all backed by Supabase.
"""
import sys
import os
import time
import logging
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NIFTY50_SYMBOL, NIFTY50_STOCKS, LIVE_REFRESH_SECONDS,
    RSI_OVERBOUGHT, RSI_OVERSOLD
)
from core.data_fetcher import DataFetcher
from core.candlestick_analyzer import CandlestickAnalyzer
from core.indicators import TechnicalIndicators
from core.strategy import StrategyEngine, SignalType
from core.ml_predictor import MLPredictor
from core.claude_analyst import ClaudeAnalyst
from broker.executor import TradeExecutor
from utils.database import SupabaseDB

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty50 Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Cached Resources ────────────────────────────────────────────────────────
@st.cache_resource
def get_fetcher():
    return DataFetcher()

@st.cache_resource
def get_db():
    return SupabaseDB()

@st.cache_resource
def get_ml():
    return MLPredictor()

def get_claude():
    """Create ClaudeAnalyst (no cache — retries if secrets become available)."""
    try:
        return ClaudeAnalyst()
    except Exception as e:
        logger.warning(f"Claude init failed: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_data(symbol, period, interval):
    fetcher = get_fetcher()
    if symbol == NIFTY50_SYMBOL:
        return fetcher.get_nifty50_history(period, interval)
    else:
        return fetcher.get_stock_history(symbol, period, interval)

@st.cache_data(ttl=30)
def fetch_live_quote(symbol):
    fetcher = get_fetcher()
    return fetcher.get_live_quote(symbol)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.sidebar.title("📊 Nifty50 Trader")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📈 Analysis", "🤖 ML Predictions",
     "🧠 Claude AI Analyst",
     "💹 Signals & Trades", "🏦 Broker & Portfolio", "⚙️ Settings"],
    index=0,
)

st.sidebar.markdown("---")
symbol_options = ["^NSEI (Nifty 50)"] + [s.replace(".NS", "") for s in NIFTY50_STOCKS[:20]]
selected_display = st.sidebar.selectbox("Symbol", symbol_options)
if selected_display.startswith("^NSEI"):
    selected_symbol = NIFTY50_SYMBOL
else:
    selected_symbol = selected_display + ".NS"

period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Interval", ["5m", "15m", "1h", "1d", "1wk"], index=3)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def enrich_dataframe(df):
    """Add all indicators + candlestick patterns to DataFrame."""
    ti = TechnicalIndicators(df)
    df = ti.add_all()
    ca = CandlestickAnalyzer(df)
    df = ca.analyze_all()
    return df


def plot_candlestick(df, title="", show_volume=True, show_indicators=True):
    """Create an interactive candlestick chart with indicators."""
    rows = 3 if show_volume else 2
    heights = [0.6, 0.2, 0.2] if show_volume else [0.7, 0.3]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.03,
        subplot_titles=[title, "Volume", "RSI"] if show_volume else [title, "RSI"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if show_indicators:
        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"],
                                     line=dict(color="orange", width=1),
                                     name="SMA 20"), row=1, col=1)
        if "sma_50" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"],
                                     line=dict(color="blue", width=1),
                                     name="SMA 50"), row=1, col=1)
        if "bb_upper" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"],
                                     line=dict(color="gray", width=0.5, dash="dash"),
                                     name="BB Upper", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"],
                                     line=dict(color="gray", width=0.5, dash="dash"),
                                     fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
                                     name="BB Lower", showlegend=False), row=1, col=1)
        if "supertrend" in df.columns:
            colors = ["#26a69a" if d == 1 else "#ef5350" for d in df.get("supertrend_dir", [1]*len(df))]
            fig.add_trace(go.Scatter(x=df["date"], y=df["supertrend"],
                                     mode="lines", name="SuperTrend",
                                     line=dict(width=1.5),
                                     marker=dict(color=colors)), row=1, col=1)

    # Volume
    if show_volume and "volume" in df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume",
                             marker_color=colors, opacity=0.5),
                      row=2, col=1)

    # RSI
    rsi_row = 3 if show_volume else 2
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"],
                                 line=dict(color="purple", width=1),
                                 name="RSI"), row=rsi_row, col=1)
        fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red",
                      opacity=0.5, row=rsi_row, col=1)
        fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green",
                      opacity=0.5, row=rsi_row, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        template="plotly_dark", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=80, b=30),
    )
    return fig


def signal_color(signal_type):
    colors = {
        "STRONG_BUY": "🟢🟢", "BUY": "🟢",
        "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴",
    }
    return colors.get(signal_type, "⚪")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Dashboard":
    st.title("🏠 Market Dashboard")

    # Live quote
    quote = fetch_live_quote(selected_symbol)
    if quote:
        col1, col2, col3, col4, col5 = st.columns(5)
        ltp = quote.get("last_price", 0)
        prev = quote.get("prev_close", ltp)
        change = ltp - prev if ltp and prev else 0
        change_pct = (change / prev * 100) if prev else 0

        col1.metric("Last Price", f"₹{ltp:,.2f}" if ltp else "N/A",
                     f"{change:+,.2f} ({change_pct:+.2f}%)" if ltp else None)
        col2.metric("Open", f"₹{quote.get('open', 0):,.2f}")
        col3.metric("High", f"₹{quote.get('high', 0):,.2f}")
        col4.metric("Low", f"₹{quote.get('low', 0):,.2f}")
        col5.metric("Volume", f"{quote.get('volume', 0):,}")

    st.markdown("---")

    # Fetch & enrich data
    with st.spinner("Fetching market data..."):
        df = fetch_data(selected_symbol, period, interval)

    if df is not None and not df.empty:
        df = enrich_dataframe(df)

        # Chart
        name = selected_display.split(" (")[0] if "(" in selected_display else selected_display
        fig = plot_candlestick(df, title=f"{name} — {interval} candles")
        st.plotly_chart(fig, use_container_width=True)

        # Quick signal
        engine = StrategyEngine(df, symbol=selected_symbol)
        signal = engine.generate_signal()
        st.markdown("### Current Signal")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**{signal_color(signal.signal.value)} {signal.signal.value}**")
        col2.markdown(f"**Confidence:** {signal.confidence:.1%}")
        col3.markdown(f"**Score:** {signal.indicator_score:+.1f}")

        with st.expander("Signal Reasons"):
            for r in signal.reasons:
                st.write(f"• {r}")
    else:
        st.warning("No data available. Check symbol or try again.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📈 Analysis":
    st.title("📈 Technical Analysis")

    with st.spinner("Loading data..."):
        df = fetch_data(selected_symbol, period, interval)

    if df is not None and not df.empty:
        df = enrich_dataframe(df)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Indicators", "🕯️ Candlestick Patterns", "📐 Fibonacci", "🔀 Multi-Timeframe"]
        )

        with tab1:
            st.subheader("Indicator Values (Latest)")
            latest = df.iloc[-1]
            cols = st.columns(4)
            indicators_display = {
                "RSI": f"{latest.get('rsi', 'N/A'):.1f}" if pd.notna(latest.get('rsi')) else "N/A",
                "MACD": f"{latest.get('macd', 'N/A'):.2f}" if pd.notna(latest.get('macd')) else "N/A",
                "ADX": f"{latest.get('adx', 'N/A'):.1f}" if pd.notna(latest.get('adx')) else "N/A",
                "ATR": f"{latest.get('atr', 'N/A'):.2f}" if pd.notna(latest.get('atr')) else "N/A",
                "SMA 20": f"₹{latest.get('sma_20', 0):,.2f}" if pd.notna(latest.get('sma_20')) else "N/A",
                "SMA 50": f"₹{latest.get('sma_50', 0):,.2f}" if pd.notna(latest.get('sma_50')) else "N/A",
                "BB Upper": f"₹{latest.get('bb_upper', 0):,.2f}" if pd.notna(latest.get('bb_upper')) else "N/A",
                "BB Lower": f"₹{latest.get('bb_lower', 0):,.2f}" if pd.notna(latest.get('bb_lower')) else "N/A",
                "SuperTrend": "Bullish 🟢" if latest.get('supertrend_dir') == 1 else "Bearish 🔴",
                "Stoch %K": f"{latest.get('stoch_k', 'N/A'):.1f}" if pd.notna(latest.get('stoch_k')) else "N/A",
                "OBV": f"{latest.get('obv', 0):,.0f}" if pd.notna(latest.get('obv')) else "N/A",
                "VWAP": f"₹{latest.get('vwap', 0):,.2f}" if pd.notna(latest.get('vwap')) else "N/A",
            }
            for i, (name, val) in enumerate(indicators_display.items()):
                cols[i % 4].metric(name, val)

            # MACD chart
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD",
                                          line=dict(color="blue")))
            macd_fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal",
                                          line=dict(color="orange")))
            macd_fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="Histogram",
                                      marker_color=["green" if v > 0 else "red"
                                                     for v in df["macd_hist"].fillna(0)]))
            macd_fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(macd_fig, use_container_width=True)

        with tab2:
            st.subheader("Detected Candlestick Patterns")
            pattern_df = df[df["patterns"].apply(lambda x: len(x) > 0)].tail(20)
            if not pattern_df.empty:
                for _, row in pattern_df.iterrows():
                    score_emoji = "🟢" if row["pattern_score"] > 0 else "🔴" if row["pattern_score"] < 0 else "🟡"
                    st.write(
                        f"{score_emoji} **{str(row.get('date', ''))[:10]}** — "
                        f"{', '.join(row['patterns'])} "
                        f"(Score: {row['pattern_score']:+d} | Close: ₹{row['close']:,.2f})"
                    )
            else:
                st.info("No notable patterns detected in this range.")

        with tab3:
            st.subheader("Fibonacci Retracement Levels")
            ti = TechnicalIndicators(df)
            fib = ti.fibonacci_levels()
            for level, price in fib.items():
                st.write(f"**{level}:** ₹{price:,.2f}")

            st.subheader("Support & Resistance")
            sr = ti.support_resistance()
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Supports:**")
                for s in sr.get("supports", []):
                    st.write(f"  ₹{s:,.2f}")
            with col2:
                st.write("**Resistances:**")
                for r in sr.get("resistances", []):
                    st.write(f"  ₹{r:,.2f}")

        with tab4:
            st.subheader("Multi-Timeframe Analysis")
            fetcher = get_fetcher()
            mtf = fetcher.get_multi_timeframe(selected_symbol)
            for tf_name, tf_df in mtf.items():
                with st.expander(f"⏱️ {tf_name.upper()} ({len(tf_df)} candles)"):
                    tf_enriched = enrich_dataframe(tf_df)
                    eng = StrategyEngine(tf_enriched, selected_symbol)
                    sig = eng.generate_signal()
                    st.markdown(
                        f"{signal_color(sig.signal.value)} **{sig.signal.value}** "
                        f"| Confidence: {sig.confidence:.1%} | Score: {sig.indicator_score:+.1f}"
                    )
    else:
        st.warning("No data available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: ML PREDICTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🤖 ML Predictions":
    st.title("🤖 ML Price Prediction")

    with st.spinner("Loading data..."):
        df = fetch_data(selected_symbol, "1y", "1d")

    if df is not None and not df.empty:
        df = enrich_dataframe(df)
        ml = get_ml()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Train Model", type="primary"):
                with st.spinner("Training ML model (this may take a moment)..."):
                    metrics = ml.train(df)
                if "error" in metrics:
                    st.error(metrics["error"])
                else:
                    st.success("Model trained successfully!")
                    st.json(metrics)

        with col2:
            if st.button("🔮 Predict Next Day"):
                pred = ml.predict(df)
                if pred is not None:
                    direction = "📈 UP" if pred > 0.5 else "📉 DOWN"
                    st.metric("Prediction", direction, f"{pred:.1%} confidence")
                else:
                    st.warning("Train the model first.")

        # Feature importance
        if ml.is_trained:
            st.subheader("Feature Importance")
            imp = ml.get_feature_importance()
            if imp is not None:
                fig = go.Figure(go.Bar(
                    x=imp["importance"].values[:15],
                    y=imp["feature"].values[:15],
                    orientation="h",
                    marker_color="steelblue"
                ))
                fig.update_layout(
                    height=400, template="plotly_dark",
                    yaxis=dict(autorange="reversed"),
                    title="Top 15 Features"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: CLAUDE AI ANALYST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🧠 Claude AI Analyst":
    st.title("🧠 Claude AI Market Analyst")

    claude = get_claude()
    if claude is None:
        st.error(
            "Anthropic API key not set. Add ANTHROPIC_API_KEY to "
            "**Streamlit Secrets** (Settings → Secrets) or your local .env file."
        )
    else:
        st.success("Claude connected ✅")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Market Analysis", "🎯 Stock Prediction",
            "💼 Investment Strategy", "💬 Ask Claude"
        ])

        # Fetch data for all tabs
        with st.spinner("Loading market data..."):
            df = fetch_data(selected_symbol, period, interval)

        if df is not None and not df.empty:
            df = enrich_dataframe(df)
            engine = StrategyEngine(df, selected_symbol)
            signal = engine.generate_signal()
            quote = fetch_live_quote(selected_symbol)

            # ── Tab 1: Full Market Analysis ──
            with tab1:
                st.subheader(f"Comprehensive {selected_display} Analysis")
                if st.button("🧠 Run Claude Analysis", type="primary", key="claude_analysis"):
                    with st.spinner("Claude is analyzing the market..."):
                        analysis = claude.analyze_market(
                            df=df, signal=signal,
                            symbol=selected_display, quote=quote
                        )
                    st.markdown(analysis)

                    # Save to session for reference
                    st.session_state["last_analysis"] = analysis

                if "last_analysis" in st.session_state:
                    with st.expander("📋 Previous Analysis"):
                        st.markdown(st.session_state["last_analysis"])

            # ── Tab 2: Quick Stock Prediction ──
            with tab2:
                st.subheader("Quick Stock Prediction")
                pred_col1, pred_col2 = st.columns([2, 1])
                with pred_col1:
                    pred_symbol = st.selectbox(
                        "Select stock",
                        ["^NSEI (Nifty 50)"] + [s.replace(".NS", "") for s in NIFTY50_STOCKS[:20]],
                        key="pred_symbol"
                    )
                with pred_col2:
                    st.write("")  # spacer
                    run_pred = st.button("🔮 Predict", type="primary", key="claude_predict")

                if run_pred:
                    # Fetch data for selected stock
                    if pred_symbol.startswith("^NSEI"):
                        pred_sym = NIFTY50_SYMBOL
                    else:
                        pred_sym = pred_symbol + ".NS"

                    with st.spinner(f"Claude is predicting {pred_symbol}..."):
                        pred_df = fetch_data(pred_sym, "6mo", "1d")
                        if pred_df is not None and not pred_df.empty:
                            pred_df = enrich_dataframe(pred_df)
                            pred_engine = StrategyEngine(pred_df, pred_symbol)
                            pred_signal = pred_engine.generate_signal()
                            prediction = claude.predict_stock(
                                df=pred_df, signal=pred_signal, symbol=pred_symbol
                            )
                            st.markdown(prediction)
                        else:
                            st.warning("No data available for this stock.")

                # Batch predict top stocks
                st.markdown("---")
                st.subheader("🔍 Batch: Top 10 Nifty 50 Predictions")
                if st.button("🚀 Scan & Predict Top 10", key="batch_predict"):
                    scanner_results = []
                    progress = st.progress(0)
                    stocks = NIFTY50_STOCKS[:10]
                    for i, stock in enumerate(stocks):
                        sdf = fetch_data(stock, "6mo", "1d")
                        if sdf is not None and not sdf.empty:
                            sdf = enrich_dataframe(sdf)
                            eng = StrategyEngine(sdf, stock)
                            sig = eng.generate_signal()
                            scanner_results.append({
                                "Symbol": stock.replace(".NS", ""),
                                "Signal": sig.signal.value,
                                "Confidence": f"{sig.confidence:.0%}",
                                "Score": sig.indicator_score,
                                "Price": f"₹{sig.entry_price:,.2f}",
                                "SL": f"₹{sig.stop_loss:,.2f}",
                                "TP": f"₹{sig.take_profit:,.2f}",
                            })
                        progress.progress((i + 1) / len(stocks))

                    if scanner_results:
                        st.dataframe(pd.DataFrame(scanner_results), use_container_width=True)
                        with st.spinner("Claude is analyzing the scanner results..."):
                            analysis = claude.nifty50_scanner_analysis(scanner_results)
                        st.markdown("### Claude's Pick")
                        st.markdown(analysis)

            # ── Tab 3: Investment Strategy ──
            with tab3:
                st.subheader("AI-Powered Investment Strategy")

                strat_col1, strat_col2 = st.columns(2)
                capital = strat_col1.number_input("Your Capital (₹)", value=100000, step=10000, key="strat_capital")
                positions = strat_col2.number_input("Current Open Positions", value=0, min_value=0, key="strat_pos")

                if st.button("📋 Generate Strategy", type="primary", key="gen_strategy"):
                    # First scan stocks for context
                    stock_sigs = []
                    with st.spinner("Scanning stocks..."):
                        for stock in NIFTY50_STOCKS[:10]:
                            sdf = fetch_data(stock, "6mo", "1d")
                            if sdf is not None and not sdf.empty:
                                sdf = enrich_dataframe(sdf)
                                eng = StrategyEngine(sdf, stock)
                                sig = eng.generate_signal()
                                stock_sigs.append({
                                    "symbol": stock.replace(".NS", ""),
                                    "signal": sig.signal.value,
                                    "confidence": f"{sig.confidence:.0%}",
                                    "price": f"{sig.entry_price:,.2f}",
                                    "score": sig.indicator_score,
                                })

                    portfolio_info = {
                        "capital": capital,
                        "invested": 0,
                        "cash": capital,
                        "positions": positions,
                    }

                    with st.spinner("Claude is building your investment strategy..."):
                        strategy = claude.investment_strategy(
                            nifty_df=df,
                            nifty_signal=signal,
                            stock_signals=stock_sigs,
                            portfolio=portfolio_info,
                        )
                    st.markdown(strategy)

            # ── Tab 4: Ask Claude ──
            with tab4:
                st.subheader("Ask Claude About the Market")
                question = st.text_area(
                    "Ask any market question:",
                    placeholder="e.g., Should I buy Reliance right now? What's the support level for Nifty?",
                    key="claude_question"
                )
                if st.button("🗣️ Ask", type="primary", key="ask_claude") and question:
                    with st.spinner("Claude is thinking..."):
                        answer = claude.ask_market_question(question, df, signal)
                    st.markdown(answer)

                # Quick explain signal
                st.markdown("---")
                st.subheader("📖 Explain Current Signal")
                if st.button("Explain in simple terms", key="explain_signal"):
                    with st.spinner("Simplifying..."):
                        explanation = claude.explain_signal(signal)
                    st.info(explanation)

        else:
            st.warning("No market data available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: SIGNALS & TRADES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "💹 Signals & Trades":
    st.title("💹 Trading Signals & Execution")

    db = get_db()

    tab1, tab2, tab3 = st.tabs(["🎯 Generate Signals", "📋 Signal History", "📊 Trade History"])

    with tab1:
        st.subheader("Generate & Execute Signals")
        with st.spinner("Analyzing..."):
            df = fetch_data(selected_symbol, period, interval)

        if df is not None and not df.empty:
            df = enrich_dataframe(df)

            # ML prediction
            ml = get_ml()
            ml_pred = ml.predict(df) if ml.is_trained else None

            engine = StrategyEngine(df, selected_symbol)
            signal = engine.generate_signal(ml_prediction=ml_pred)

            # Signal display
            st.markdown(f"### {signal_color(signal.signal.value)} {signal.signal.value}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Entry", f"₹{signal.entry_price:,.2f}")
            col2.metric("Stop Loss", f"₹{signal.stop_loss:,.2f}")
            col3.metric("Take Profit", f"₹{signal.take_profit:,.2f}")
            col4.metric("Confidence", f"{signal.confidence:.1%}")

            if ml_pred:
                st.info(f"ML Prediction: {'📈 UP' if ml_pred > 0.5 else '📉 DOWN'} ({ml_pred:.1%})")

            with st.expander("📝 Signal Reasons"):
                for r in signal.reasons:
                    st.write(f"• {r}")

            st.markdown("---")

            # Execution controls
            col1, col2, col3 = st.columns(3)
            broker_choice = col1.selectbox("Broker", ["zerodha", "angelone", "groww"])
            auto_trade = col2.checkbox("Auto-trade enabled", value=False)
            dry_run = col3.checkbox("Dry run (simulate)", value=True)

            if st.button("🚀 Execute Signal", type="primary"):
                executor = TradeExecutor(
                    broker_name=broker_choice,
                    auto_trade=auto_trade,
                )
                result = executor.execute_signal(signal, dry_run=dry_run)
                if result.get("order_placed"):
                    st.success(f"Order placed! ID: {result['order_id']}")
                elif result.get("actionable"):
                    st.info(f"Signal actionable — {result.get('reason', 'Dry run')}")
                    st.json(result["position_size"])
                else:
                    st.warning(result.get("reason", "Signal not actionable"))

            # Scan multiple stocks
            st.markdown("---")
            st.subheader("🔍 Nifty50 Stock Scanner")
            if st.button("Scan Top 10 Stocks"):
                scanner_results = []
                progress = st.progress(0)
                stocks = NIFTY50_STOCKS[:10]
                for i, stock in enumerate(stocks):
                    sdf = fetch_data(stock, "6mo", "1d")
                    if sdf is not None and not sdf.empty:
                        sdf = enrich_dataframe(sdf)
                        eng = StrategyEngine(sdf, stock)
                        sig = eng.generate_signal()
                        scanner_results.append({
                            "Symbol": stock.replace(".NS", ""),
                            "Signal": sig.signal.value,
                            "Confidence": f"{sig.confidence:.0%}",
                            "Score": sig.indicator_score,
                            "Price": f"₹{sig.entry_price:,.2f}",
                            "SL": f"₹{sig.stop_loss:,.2f}",
                            "TP": f"₹{sig.take_profit:,.2f}",
                        })
                    progress.progress((i + 1) / len(stocks))

                if scanner_results:
                    scan_df = pd.DataFrame(scanner_results)
                    scan_df = scan_df.sort_values("Score", ascending=False)
                    st.dataframe(scan_df, use_container_width=True)

    with tab2:
        st.subheader("Recent Signals (from Supabase)")
        if db.is_connected:
            signals = db.get_recent_signals(limit=30, symbol=None)
            if signals:
                sig_df = pd.DataFrame(signals)
                st.dataframe(sig_df, use_container_width=True)
            else:
                st.info("No signals stored yet. Generate signals to populate.")
        else:
            st.warning("Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY in .env")

    with tab3:
        st.subheader("Trade History (from Supabase)")
        if db.is_connected:
            trades = db.get_trade_history(limit=50)
            if trades:
                trade_df = pd.DataFrame(trades)
                st.dataframe(trade_df, use_container_width=True)

                # P&L summary
                closed = [t for t in trades if t.get("status") == "CLOSED"]
                if closed:
                    total_pnl = sum(t.get("pnl", 0) for t in closed)
                    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
                    win_rate = wins / len(closed) * 100
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total P&L", f"₹{total_pnl:,.2f}")
                    col2.metric("Trades", len(closed))
                    col3.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.info("No trades found.")
        else:
            st.warning("Supabase not configured.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: BROKER & PORTFOLIO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🏦 Broker & Portfolio":
    st.title("🏦 Broker & Portfolio")

    db = get_db()

    tab1, tab2, tab3 = st.tabs(["💰 Account", "📈 Portfolio History", "👁️ Watchlist"])

    with tab1:
        broker_choice = st.selectbox("Select Broker", ["zerodha", "angelone", "groww"])
        if st.button("🔗 Connect & Fetch"):
            executor = TradeExecutor(broker_name=broker_choice)
            if executor.connect_broker():
                st.success(f"Connected to {broker_choice}!")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Balance")
                    balance = executor.broker.get_balance()
                    if balance:
                        st.json(balance)

                with col2:
                    st.subheader("Positions")
                    positions = executor.broker.get_positions()
                    if positions:
                        st.dataframe(pd.DataFrame(positions))
                    else:
                        st.info("No open positions")

                st.subheader("Holdings")
                holdings = executor.broker.get_holdings()
                if holdings:
                    st.dataframe(pd.DataFrame(holdings))
                else:
                    st.info("No holdings")
            else:
                st.error(f"Failed to connect to {broker_choice}. Check your API keys in .env")

    with tab2:
        st.subheader("Portfolio Value Over Time")
        if db.is_connected:
            snapshots = db.get_portfolio_history(30)
            if snapshots:
                snap_df = pd.DataFrame(snapshots)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=snap_df["snapshot_at"], y=snap_df["total_capital"],
                    mode="lines+markers", name="Total Capital",
                    line=dict(color="#26a69a", width=2)
                ))
                fig.update_layout(height=400, template="plotly_dark",
                                  title="Portfolio Value History")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No snapshots yet. They'll appear once you start trading.")
        else:
            st.warning("Supabase not configured.")

        # Performance metrics
        if db.is_connected:
            perf = db.get_performance_history(30)
            if perf:
                st.subheader("Strategy Performance")
                perf_df = pd.DataFrame(perf)
                st.dataframe(perf_df, use_container_width=True)

    with tab3:
        st.subheader("Watchlist")
        if db.is_connected:
            # Add to watchlist
            new_symbol = st.text_input("Add symbol to watchlist (e.g. RELIANCE.NS)")
            if st.button("➕ Add") and new_symbol:
                db.add_to_watchlist(new_symbol)
                st.success(f"Added {new_symbol}")
                st.rerun()

            # Display watchlist
            watchlist = db.get_watchlist()
            if watchlist:
                for item in watchlist:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    col1.write(item["symbol"])
                    quote = fetch_live_quote(item["symbol"])
                    col2.write(f"₹{quote.get('last_price', 'N/A')}" if quote else "N/A")
                    if col3.button("❌", key=f"rm_{item['symbol']}"):
                        db.remove_from_watchlist(item["symbol"])
                        st.rerun()
            else:
                st.info("Watchlist is empty.")
        else:
            st.warning("Supabase not configured.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    db = get_db()

    st.subheader("Trading Settings")

    if db.is_connected:
        settings = db.get_all_settings()
    else:
        settings = {}

    capital = st.number_input(
        "Trading Capital (₹)", value=float(settings.get("capital", 100000)),
        min_value=10000.0, step=10000.0
    )
    risk_pct = st.slider(
        "Risk per Trade (%)", 1, 10,
        int(float(settings.get("risk_per_trade", 0.02)) * 100)
    ) / 100
    max_pos = st.number_input(
        "Max Open Positions", value=int(settings.get("max_positions", 5)),
        min_value=1, max_value=20
    )
    auto_trade = st.checkbox(
        "Enable Auto-Trading (⚠️ real orders!)",
        value=bool(settings.get("auto_trade_enabled", False))
    )
    default_broker = st.selectbox(
        "Default Broker",
        ["zerodha", "angelone", "groww"],
        index=["zerodha", "angelone", "groww"].index(
            settings.get("default_broker", "zerodha")
        ) if settings.get("default_broker") in ["zerodha", "angelone", "groww"] else 0
    )

    if st.button("💾 Save Settings"):
        if db.is_connected:
            db.set_setting("capital", capital)
            db.set_setting("risk_per_trade", risk_pct)
            db.set_setting("max_positions", max_pos)
            db.set_setting("auto_trade_enabled", auto_trade)
            db.set_setting("default_broker", default_broker)
            st.success("Settings saved to Supabase!")
        else:
            st.warning("Settings not saved — Supabase not configured. They'll work in-memory for this session.")

    st.markdown("---")
    st.subheader("API Keys Status")
    from config import (
        ZERODHA_API_KEY, ANGEL_API_KEY, SUPABASE_URL, ANTHROPIC_API_KEY
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zerodha", "✅ Set" if ZERODHA_API_KEY else "❌ Not set")
    col2.metric("Angel One", "✅ Set" if ANGEL_API_KEY else "❌ Not set")
    col3.metric("Supabase", "✅ Connected" if SUPABASE_URL else "❌ Not set")
    col4.metric("Claude AI", "✅ Set" if ANTHROPIC_API_KEY else "❌ Not set")

    st.markdown("---")
    st.subheader("Database Status")
    if db.is_connected:
        st.success("Supabase connected ✅")
        st.caption("All signals, trades, and settings are persisted in Supabase.")
    else:
        st.error("Supabase not connected ❌")
        st.markdown("""
        **Setup instructions:**
        1. Create a project at [supabase.com](https://supabase.com)
        2. Run the SQL schema from `utils/supabase_schema.sql` in the SQL Editor
        3. Copy your project URL and anon key to `.env`:
        ```
        SUPABASE_URL=https://your-project.supabase.co
        SUPABASE_KEY=your_anon_key
        ```
        4. Restart the app
        """)

