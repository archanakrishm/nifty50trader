"""
Supabase Database Layer
Handles all database operations — signals, trades, portfolio, settings.
"""
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Wrapper around Supabase client for all DB operations."""

    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("Supabase credentials not set — DB features disabled")
            self.client: Optional[Client] = None
        else:
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client initialized")

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    def _check(self):
        if not self.is_connected:
            raise ConnectionError("Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY in .env")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TRADE SIGNALS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def save_signal(self, signal_dict: Dict) -> Optional[Dict]:
        """Store a generated trade signal."""
        self._check()
        try:
            result = (
                self.client.table("trade_signals")
                .insert(signal_dict)
                .execute()
            )
            logger.info(f"Signal saved: {signal_dict.get('symbol')} {signal_dict.get('signal')}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return None

    def save_signals_batch(self, signals: List[Dict]) -> int:
        """Bulk-insert signals."""
        self._check()
        try:
            result = (
                self.client.table("trade_signals")
                .insert(signals)
                .execute()
            )
            count = len(result.data) if result.data else 0
            logger.info(f"Saved {count} signals in batch")
            return count
        except Exception as e:
            logger.error(f"Batch signal save error: {e}")
            return 0

    def get_recent_signals(self, limit: int = 50, symbol: Optional[str] = None) -> List[Dict]:
        """Retrieve recent signals, optionally filtered by symbol."""
        self._check()
        try:
            query = (
                self.client.table("trade_signals")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
            )
            if symbol:
                query = query.eq("symbol", symbol)
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TRADES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def save_trade(self, trade: Dict) -> Optional[Dict]:
        """Record an executed trade."""
        self._check()
        try:
            result = (
                self.client.table("trades")
                .insert(trade)
                .execute()
            )
            logger.info(f"Trade saved: {trade.get('symbol')} {trade.get('side')}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return None

    def close_trade(self, trade_id: int, exit_price: float, pnl: float, pnl_pct: float) -> bool:
        """Mark a trade as closed with P&L."""
        self._check()
        try:
            self.client.table("trades").update({
                "status": "CLOSED",
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "closed_at": datetime.now().isoformat(),
            }).eq("id", trade_id).execute()
            logger.info(f"Trade {trade_id} closed — PnL: ₹{pnl:.2f}")
            return True
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return False

    def get_open_trades(self) -> List[Dict]:
        """Get all currently open trades."""
        self._check()
        try:
            result = (
                self.client.table("trades")
                .select("*")
                .eq("status", "OPEN")
                .order("opened_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching open trades: {e}")
            return []

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        self._check()
        try:
            result = (
                self.client.table("trades")
                .select("*")
                .order("opened_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PORTFOLIO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def save_portfolio_snapshot(self, snapshot: Dict) -> Optional[Dict]:
        """Save a portfolio snapshot."""
        self._check()
        try:
            result = (
                self.client.table("portfolio_snapshots")
                .insert(snapshot)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return None

    def get_portfolio_history(self, limit: int = 30) -> List[Dict]:
        """Get recent portfolio snapshots."""
        self._check()
        try:
            result = (
                self.client.table("portfolio_snapshots")
                .select("*")
                .order("snapshot_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STRATEGY PERFORMANCE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def save_daily_performance(self, perf: Dict) -> Optional[Dict]:
        """Log daily strategy performance."""
        self._check()
        try:
            result = (
                self.client.table("strategy_performance")
                .insert(perf)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving performance: {e}")
            return None

    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get performance over recent days."""
        self._check()
        try:
            result = (
                self.client.table("strategy_performance")
                .select("*")
                .order("date", desc=True)
                .limit(days)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching performance: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  WATCHLIST
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def add_to_watchlist(self, symbol: str, notes: str = "") -> bool:
        self._check()
        try:
            self.client.table("watchlist").upsert({
                "symbol": symbol,
                "notes": notes
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Watchlist add error: {e}")
            return False

    def remove_from_watchlist(self, symbol: str) -> bool:
        self._check()
        try:
            self.client.table("watchlist").delete().eq("symbol", symbol).execute()
            return True
        except Exception as e:
            logger.error(f"Watchlist remove error: {e}")
            return False

    def get_watchlist(self) -> List[Dict]:
        self._check()
        try:
            result = (
                self.client.table("watchlist")
                .select("*")
                .order("added_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Watchlist fetch error: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def get_setting(self, key: str) -> Any:
        self._check()
        try:
            result = (
                self.client.table("app_settings")
                .select("value")
                .eq("key", key)
                .single()
                .execute()
            )
            return result.data.get("value") if result.data else None
        except Exception as e:
            logger.error(f"Settings fetch error: {e}")
            return None

    def set_setting(self, key: str, value: Any) -> bool:
        self._check()
        try:
            self.client.table("app_settings").upsert({
                "key": key,
                "value": value,
                "updated_at": datetime.now().isoformat(),
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Settings update error: {e}")
            return False

    def get_all_settings(self) -> Dict[str, Any]:
        self._check()
        try:
            result = self.client.table("app_settings").select("*").execute()
            return {row["key"]: row["value"] for row in (result.data or [])}
        except Exception as e:
            logger.error(f"Settings fetch error: {e}")
            return {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MARKET DATA CACHE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def cache_market_data(self, symbol: str, timeframe: str, rows: List[Dict]) -> int:
        """Cache OHLCV data to reduce API calls."""
        self._check()
        try:
            for row in rows:
                row["symbol"] = symbol
                row["timeframe"] = timeframe
            result = (
                self.client.table("market_data_cache")
                .upsert(rows, on_conflict="symbol,timeframe,date")
                .execute()
            )
            return len(result.data) if result.data else 0
        except Exception as e:
            logger.error(f"Cache error: {e}")
            return 0

    def get_cached_data(self, symbol: str, timeframe: str, limit: int = 500) -> List[Dict]:
        self._check()
        try:
            result = (
                self.client.table("market_data_cache")
                .select("*")
                .eq("symbol", symbol)
                .eq("timeframe", timeframe)
                .order("date", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Cache fetch error: {e}")
            return []
