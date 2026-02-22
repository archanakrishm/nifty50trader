"""
Base Broker Interface
Abstract class that all broker integrations must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List


class BaseBroker(ABC):
    """Abstract broker interface for order execution."""

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish connection."""
        ...

    @abstractmethod
    def get_balance(self) -> Dict:
        """Return available balance/margins."""
        ...

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Return current open positions."""
        ...

    @abstractmethod
    def get_holdings(self) -> List[Dict]:
        """Return delivery holdings (CNC)."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,             # "BUY" or "SELL"
        qty: int,
        order_type: str = "MARKET",   # MARKET / LIMIT / SL / SL-M
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "CNC",  # CNC (delivery) / MIS (intraday) / NRML
    ) -> Optional[str]:
        """Place an order. Returns broker order ID or None."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of a specific order."""
        ...

    @abstractmethod
    def get_order_history(self) -> List[Dict]:
        """Get today's order history."""
        ...

    @abstractmethod
    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price for a symbol."""
        ...
