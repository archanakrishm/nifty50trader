"""
Trade Executor
Bridges strategy signals with broker execution, logs to Supabase.
"""
import logging
from typing import Optional, Dict

from broker.base import BaseBroker
from broker.zerodha_broker import ZerodhaBroker
from broker.angelone_broker import AngelOneBroker
from core.strategy import TradeSignal, SignalType, PortfolioManager
from utils.database import SupabaseDB

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Orchestrates signal → order → DB logging."""

    def __init__(
        self,
        broker_name: str = "zerodha",
        capital: float = 100000.0,
        auto_trade: bool = False,
    ):
        self.broker_name = broker_name
        self.broker: Optional[BaseBroker] = None
        self.db = SupabaseDB()
        self.portfolio = PortfolioManager(capital=capital)
        self.auto_trade = auto_trade

    def connect_broker(self) -> bool:
        """Initialize and connect to the chosen broker."""
        if self.broker_name == "zerodha":
            self.broker = ZerodhaBroker()
        elif self.broker_name in ("angelone", "groww"):
            self.broker = AngelOneBroker()
        else:
            logger.error(f"Unknown broker: {self.broker_name}")
            return False
        return self.broker.connect()

    def execute_signal(self, signal: TradeSignal, dry_run: bool = True) -> Dict:
        """
        Process a signal:
          1. Save signal to DB
          2. Check if it's actionable
          3. Calculate position size
          4. Place order (if auto_trade and not dry_run)
          5. Log trade to DB

        Args:
            signal: Trade signal to execute.
            dry_run: If True, simulate only (no real order).

        Returns:
            Execution report dict.
        """
        report = {
            "signal": signal.to_dict(),
            "actionable": False,
            "position_size": {},
            "order_placed": False,
            "order_id": None,
            "dry_run": dry_run,
        }

        # 1. Save signal to DB
        if self.db.is_connected:
            self.db.save_signal(signal.to_dict())

        # 2. Check if actionable
        if not self.portfolio.should_execute(signal):
            report["reason"] = "Signal not strong enough to trade"
            logger.info(f"Signal skipped: {signal.symbol} {signal.signal.value} (conf={signal.confidence})")
            return report

        report["actionable"] = True

        # 3. Position sizing
        sizing = self.portfolio.calculate_position_size(signal)
        report["position_size"] = sizing
        if sizing["qty"] == 0:
            report["reason"] = sizing.get("reason", "Zero quantity")
            return report

        # 4. Execute order
        if self.auto_trade and not dry_run:
            if self.broker is None or not self.broker.connected if hasattr(self.broker, 'connected') else True:
                if not self.connect_broker():
                    report["reason"] = "Broker connection failed"
                    return report

            side = "BUY" if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY) else "SELL"
            order_id = self.broker.place_order(
                symbol=signal.symbol,
                side=side,
                qty=sizing["qty"],
                order_type="MARKET",
                product="CNC",
            )

            if order_id:
                report["order_placed"] = True
                report["order_id"] = order_id

                # 5. Log to DB
                if self.db.is_connected:
                    self.db.save_trade({
                        "symbol": signal.symbol,
                        "side": side,
                        "qty": sizing["qty"],
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "broker": self.broker_name,
                        "broker_order_id": order_id,
                        "status": "OPEN",
                    })
                logger.info(f"Order executed: {order_id} | {side} {sizing['qty']} {signal.symbol}")
            else:
                report["reason"] = "Order placement failed"
        else:
            side = "BUY" if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY) else "SELL"
            logger.info(
                f"[DRY RUN] Would {side} {sizing['qty']} {signal.symbol} "
                f"@ ₹{signal.entry_price} | SL: ₹{signal.stop_loss} | TP: ₹{signal.take_profit}"
            )
            report["reason"] = "Dry run — no order placed"

        return report

    def check_stop_losses(self) -> list:
        """Check open trades for stop loss / take profit triggers."""
        if not self.db.is_connected:
            return []

        triggered = []
        open_trades = self.db.get_open_trades()

        for trade in open_trades:
            symbol = trade["symbol"]
            ltp = None

            if self.broker and hasattr(self.broker, 'get_ltp'):
                ltp = self.broker.get_ltp(symbol)

            if ltp is None:
                continue

            entry = trade["entry_price"]
            sl = trade.get("stop_loss")
            tp = trade.get("take_profit")
            side = trade["side"]

            should_close = False
            reason = ""

            if side == "BUY":
                if sl and ltp <= sl:
                    should_close = True
                    reason = "Stop loss hit"
                elif tp and ltp >= tp:
                    should_close = True
                    reason = "Take profit hit"
            elif side == "SELL":
                if sl and ltp >= sl:
                    should_close = True
                    reason = "Stop loss hit"
                elif tp and ltp <= tp:
                    should_close = True
                    reason = "Take profit hit"

            if should_close:
                pnl = (ltp - entry) * trade["qty"] if side == "BUY" else (entry - ltp) * trade["qty"]
                pnl_pct = pnl / (entry * trade["qty"]) if entry > 0 else 0

                self.db.close_trade(trade["id"], ltp, pnl, pnl_pct)
                triggered.append({
                    "trade_id": trade["id"],
                    "symbol": symbol,
                    "reason": reason,
                    "pnl": round(pnl, 2),
                    "ltp": ltp,
                })
                logger.info(f"Trade {trade['id']} closed: {reason} | PnL: ₹{pnl:.2f}")

        return triggered
