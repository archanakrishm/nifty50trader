"""
Zerodha Kite Connect Broker Integration
Uses kiteconnect SDK for order placement and account management.
"""
import logging
from typing import Dict, Optional, List

from broker.base import BaseBroker
from config import ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN

logger = logging.getLogger(__name__)


class ZerodhaBroker(BaseBroker):
    """Zerodha Kite Connect integration."""

    # NSE symbol → Kite exchange token mapping
    EXCHANGE = "NSE"
    PRODUCT_MAP = {
        "CNC": "CNC",
        "MIS": "MIS",
        "NRML": "NRML",
    }
    ORDER_TYPE_MAP = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "SL",
        "SL-M": "SL-M",
    }

    def __init__(self):
        self.kite = None
        self.connected = False

    def connect(self) -> bool:
        """Connect using pre-generated access token."""
        try:
            from kiteconnect import KiteConnect
            if not ZERODHA_API_KEY or not ZERODHA_ACCESS_TOKEN:
                logger.error("Zerodha API key or access token not set")
                return False

            self.kite = KiteConnect(api_key=ZERODHA_API_KEY)
            self.kite.set_access_token(ZERODHA_ACCESS_TOKEN)

            # Validate session
            profile = self.kite.profile()
            logger.info(f"Zerodha connected: {profile.get('user_name', 'Unknown')}")
            self.connected = True
            return True
        except ImportError:
            logger.error("kiteconnect package not installed. pip install kiteconnect")
            return False
        except Exception as e:
            logger.error(f"Zerodha connection failed: {e}")
            return False

    def get_balance(self) -> Dict:
        if not self.connected:
            return {}
        try:
            margins = self.kite.margins()
            equity = margins.get("equity", {})
            return {
                "available_cash": equity.get("available", {}).get("cash", 0),
                "used_margin": equity.get("utilised", {}).get("debits", 0),
                "total": equity.get("net", 0),
            }
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            pos = self.kite.positions()
            return [
                {
                    "symbol": p["tradingsymbol"],
                    "qty": p["quantity"],
                    "avg_price": p["average_price"],
                    "ltp": p["last_price"],
                    "pnl": p["pnl"],
                    "product": p["product"],
                }
                for p in pos.get("net", [])
                if p["quantity"] != 0
            ]
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []

    def get_holdings(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            holdings = self.kite.holdings()
            return [
                {
                    "symbol": h["tradingsymbol"],
                    "qty": h["quantity"],
                    "avg_price": h["average_price"],
                    "ltp": h["last_price"],
                    "pnl": h["pnl"],
                    "day_change_pct": h.get("day_change_percentage", 0),
                }
                for h in holdings
            ]
        except Exception as e:
            logger.error(f"Holdings error: {e}")
            return []

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "CNC",
    ) -> Optional[str]:
        if not self.connected:
            logger.error("Not connected to Zerodha")
            return None
        try:
            transaction = self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.EXCHANGE,
                tradingsymbol=symbol.replace(".NS", ""),
                transaction_type=transaction,
                quantity=qty,
                order_type=self.ORDER_TYPE_MAP.get(order_type, "MARKET"),
                product=self.PRODUCT_MAP.get(product, "CNC"),
                price=price if order_type == "LIMIT" else None,
                trigger_price=trigger_price if order_type in ("SL", "SL-M") else None,
            )
            logger.info(f"Zerodha order placed: {order_id} | {side} {qty} {symbol}")
            return str(order_id)
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        if not self.connected:
            return False
        try:
            self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False

    def get_order_status(self, order_id: str) -> Dict:
        if not self.connected:
            return {}
        try:
            orders = self.kite.order_history(order_id)
            if orders:
                latest = orders[-1]
                return {
                    "order_id": latest["order_id"],
                    "status": latest["status"],
                    "filled_qty": latest.get("filled_quantity", 0),
                    "avg_price": latest.get("average_price", 0),
                    "symbol": latest["tradingsymbol"],
                }
            return {}
        except Exception as e:
            logger.error(f"Order status error: {e}")
            return {}

    def get_order_history(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            orders = self.kite.orders()
            return [
                {
                    "order_id": o["order_id"],
                    "symbol": o["tradingsymbol"],
                    "side": o["transaction_type"],
                    "qty": o["quantity"],
                    "price": o.get("average_price", 0),
                    "status": o["status"],
                    "order_type": o["order_type"],
                    "timestamp": str(o.get("order_timestamp", "")),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Order history error: {e}")
            return []

    def get_ltp(self, symbol: str) -> Optional[float]:
        if not self.connected:
            return None
        try:
            clean = symbol.replace(".NS", "")
            key = f"NSE:{clean}"
            data = self.kite.ltp([key])
            return data.get(key, {}).get("last_price")
        except Exception as e:
            logger.error(f"LTP error: {e}")
            return None
