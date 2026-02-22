"""
Angel One SmartAPI Broker Integration
Uses smartapi-python SDK. Also works if your demat is through Groww
(which uses Angel One as the backend broker).
"""
import logging
from typing import Dict, Optional, List

from broker.base import BaseBroker
from config import ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET

logger = logging.getLogger(__name__)


class AngelOneBroker(BaseBroker):
    """Angel One / SmartAPI integration (also powers Groww demat)."""

    EXCHANGE = "NSE"
    PRODUCT_MAP = {"CNC": "DELIVERY", "MIS": "INTRADAY", "NRML": "CARRYFORWARD"}
    ORDER_TYPE_MAP = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "STOPLOSS_LIMIT",
        "SL-M": "STOPLOSS_MARKET",
    }

    # Common Nifty50 symbol → Angel token mapping (partial; full list from API)
    SYMBOL_TOKEN_MAP = {
        "RELIANCE": "2885", "TCS": "11536", "HDFCBANK": "1333",
        "INFY": "1594", "ICICIBANK": "4963", "HINDUNILVR": "1394",
        "ITC": "1660", "SBIN": "3045", "BHARTIARTL": "10604",
        "KOTAKBANK": "1922", "LT": "11483", "AXISBANK": "5900",
    }

    def __init__(self):
        self.smart_api = None
        self.connected = False
        self._auth_token = None
        self._refresh_token = None

    def connect(self) -> bool:
        try:
            from SmartApi import SmartConnect
            import pyotp

            if not all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD]):
                logger.error("Angel One credentials not set")
                return False

            self.smart_api = SmartConnect(api_key=ANGEL_API_KEY)

            # Generate TOTP if secret is available
            totp = ""
            if ANGEL_TOTP_SECRET:
                totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()

            data = self.smart_api.generateSession(
                clientCode=ANGEL_CLIENT_ID,
                password=ANGEL_PASSWORD,
                totp=totp
            )

            if data.get("status"):
                self._auth_token = data["data"]["jwtToken"]
                self._refresh_token = data["data"]["refreshToken"]
                profile = self.smart_api.getProfile(self._refresh_token)
                name = profile.get("data", {}).get("name", "Unknown")
                logger.info(f"Angel One connected: {name}")
                self.connected = True
                return True
            else:
                logger.error(f"Angel One login failed: {data.get('message')}")
                return False
        except ImportError:
            logger.error("smartapi-python not installed. pip install smartapi-python")
            return False
        except Exception as e:
            logger.error(f"Angel One connection failed: {e}")
            return False

    def _get_token(self, symbol: str) -> str:
        """Resolve NSE symbol to Angel instrument token."""
        clean = symbol.replace(".NS", "")
        return self.SYMBOL_TOKEN_MAP.get(clean, "")

    def get_balance(self) -> Dict:
        if not self.connected:
            return {}
        try:
            rms = self.smart_api.rmsLimit()
            data = rms.get("data", {})
            return {
                "available_cash": float(data.get("availablecash", 0)),
                "used_margin": float(data.get("utiliseddebits", 0)),
                "total": float(data.get("net", 0)),
            }
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            pos = self.smart_api.position()
            data = pos.get("data", []) or []
            return [
                {
                    "symbol": p.get("tradingsymbol", ""),
                    "qty": int(p.get("netqty", 0)),
                    "avg_price": float(p.get("averageprice", 0)),
                    "ltp": float(p.get("ltp", 0)),
                    "pnl": float(p.get("pnl", 0)),
                    "product": p.get("producttype", ""),
                }
                for p in data
                if int(p.get("netqty", 0)) != 0
            ]
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []

    def get_holdings(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            holdings = self.smart_api.holding()
            data = holdings.get("data", []) or []
            return [
                {
                    "symbol": h.get("tradingsymbol", ""),
                    "qty": int(h.get("quantity", 0)),
                    "avg_price": float(h.get("averageprice", 0)),
                    "ltp": float(h.get("ltp", 0)),
                    "pnl": float(h.get("profitandloss", 0)),
                }
                for h in data
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
            logger.error("Not connected to Angel One")
            return None
        try:
            clean_symbol = symbol.replace(".NS", "")
            token = self._get_token(symbol)

            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": clean_symbol,
                "symboltoken": token,
                "transactiontype": side,
                "exchange": self.EXCHANGE,
                "ordertype": self.ORDER_TYPE_MAP.get(order_type, "MARKET"),
                "producttype": self.PRODUCT_MAP.get(product, "DELIVERY"),
                "duration": "DAY",
                "quantity": str(qty),
            }
            if order_type == "LIMIT":
                order_params["price"] = str(price)
            if order_type in ("SL", "SL-M"):
                order_params["triggerprice"] = str(trigger_price)

            result = self.smart_api.placeOrder(order_params)
            if result:
                logger.info(f"Angel One order placed: {result} | {side} {qty} {clean_symbol}")
                return str(result)
            return None
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        if not self.connected:
            return False
        try:
            result = self.smart_api.cancelOrder(order_id, "NORMAL")
            logger.info(f"Order cancelled: {order_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False

    def get_order_status(self, order_id: str) -> Dict:
        if not self.connected:
            return {}
        try:
            orders = self.smart_api.orderBook()
            data = orders.get("data", []) or []
            for o in data:
                if str(o.get("orderid")) == order_id:
                    return {
                        "order_id": o["orderid"],
                        "status": o.get("orderstatus", ""),
                        "filled_qty": int(o.get("filledshares", 0)),
                        "avg_price": float(o.get("averageprice", 0)),
                        "symbol": o.get("tradingsymbol", ""),
                    }
            return {}
        except Exception as e:
            logger.error(f"Order status error: {e}")
            return {}

    def get_order_history(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            orders = self.smart_api.orderBook()
            data = orders.get("data", []) or []
            return [
                {
                    "order_id": o.get("orderid", ""),
                    "symbol": o.get("tradingsymbol", ""),
                    "side": o.get("transactiontype", ""),
                    "qty": int(o.get("quantity", 0)),
                    "price": float(o.get("averageprice", 0)),
                    "status": o.get("orderstatus", ""),
                    "order_type": o.get("ordertype", ""),
                    "timestamp": o.get("updatetime", ""),
                }
                for o in data
            ]
        except Exception as e:
            logger.error(f"Order history error: {e}")
            return []

    def get_ltp(self, symbol: str) -> Optional[float]:
        if not self.connected:
            return None
        try:
            clean = symbol.replace(".NS", "")
            token = self._get_token(symbol)
            data = self.smart_api.ltpData(self.EXCHANGE, clean, token)
            return float(data.get("data", {}).get("ltp", 0))
        except Exception as e:
            logger.error(f"LTP error: {e}")
            return None
