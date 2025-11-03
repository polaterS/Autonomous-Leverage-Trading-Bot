"""
Advanced Order Types Module
Provides trailing stop-loss, limit orders, and OCO orders
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """
    Manages trailing stop-loss orders that follow price movements.

    Features:
    - Automatic stop-loss adjustment as price moves favorably
    - Configurable trailing distance (percentage or fixed amount)
    - Locks in profits while allowing for upside
    """

    def __init__(self):
        self.active_trailing_stops: Dict[str, Dict] = {}  # position_id -> trailing stop config

    async def enable_trailing_stop(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: Decimal,
        current_price: Decimal,
        trailing_distance_percent: Decimal = Decimal("0.02")  # 2% default
    ) -> Dict[str, Any]:
        """
        Enable trailing stop for a position.

        Args:
            position_id: Position ID
            symbol: Trading symbol
            side: LONG or SHORT
            entry_price: Entry price
            current_price: Current market price
            trailing_distance_percent: How far behind price should stop trail (default 2%)

        Returns:
            Trailing stop configuration
        """
        # Calculate initial stop-loss
        if side == "LONG":
            initial_stop = current_price * (Decimal("1") - trailing_distance_percent)
            highest_price = current_price
        else:  # SHORT
            initial_stop = current_price * (Decimal("1") + trailing_distance_percent)
            lowest_price = current_price

        config = {
            'position_id': position_id,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'trailing_distance_percent': trailing_distance_percent,
            'current_stop': initial_stop,
            'highest_price': highest_price if side == "LONG" else None,
            'lowest_price': lowest_price if side == "SHORT" else None,
            'enabled': True,
            'created_at': datetime.now()
        }

        self.active_trailing_stops[position_id] = config

        logger.info(f"✅ Trailing stop enabled for {symbol} {side}: "
                   f"Initial stop @ ${float(initial_stop):.2f} "
                   f"(trailing {float(trailing_distance_percent) * 100:.1f}%)")

        return config

    async def update_trailing_stop(
        self,
        position_id: str,
        current_price: Decimal
    ) -> Optional[Dict[str, Any]]:
        """
        Update trailing stop based on new price.

        Returns:
            Updated config if stop was adjusted, None if not adjusted or position not found
        """
        if position_id not in self.active_trailing_stops:
            return None

        config = self.active_trailing_stops[position_id]

        if not config['enabled']:
            return None

        side = config['side']
        trailing_dist = config['trailing_distance_percent']
        old_stop = config['current_stop']

        # Update trailing stop based on price movement
        if side == "LONG":
            # Update highest price seen
            if current_price > config['highest_price']:
                config['highest_price'] = current_price

                # Calculate new stop (trailing behind highest price)
                new_stop = config['highest_price'] * (Decimal("1") - trailing_dist)

                # Only move stop UP, never down
                if new_stop > old_stop:
                    config['current_stop'] = new_stop
                    logger.info(f"📈 Trailing stop adjusted UP for {config['symbol']}: "
                               f"${float(old_stop):.2f} → ${float(new_stop):.2f} "
                               f"(price: ${float(current_price):.2f})")
                    return config

        else:  # SHORT
            # Update lowest price seen
            if current_price < config['lowest_price']:
                config['lowest_price'] = current_price

                # Calculate new stop (trailing above lowest price)
                new_stop = config['lowest_price'] * (Decimal("1") + trailing_dist)

                # Only move stop DOWN, never up
                if new_stop < old_stop:
                    config['current_stop'] = new_stop
                    logger.info(f"📉 Trailing stop adjusted DOWN for {config['symbol']}: "
                               f"${float(old_stop):.2f} → ${float(new_stop):.2f} "
                               f"(price: ${float(current_price):.2f})")
                    return config

        return None  # No adjustment needed

    def should_trigger_trailing_stop(
        self,
        position_id: str,
        current_price: Decimal
    ) -> bool:
        """
        Check if trailing stop should be triggered.

        Returns:
            True if stop-loss hit, False otherwise
        """
        if position_id not in self.active_trailing_stops:
            return False

        config = self.active_trailing_stops[position_id]

        if not config['enabled']:
            return False

        side = config['side']
        stop_price = config['current_stop']

        # Check if stop triggered
        if side == "LONG":
            if current_price <= stop_price:
                logger.warning(f"🛑 Trailing stop TRIGGERED for {config['symbol']} LONG: "
                             f"Price ${float(current_price):.2f} <= Stop ${float(stop_price):.2f}")
                return True

        else:  # SHORT
            if current_price >= stop_price:
                logger.warning(f"🛑 Trailing stop TRIGGERED for {config['symbol']} SHORT: "
                             f"Price ${float(current_price):.2f} >= Stop ${float(stop_price):.2f}")
                return True

        return False

    def disable_trailing_stop(self, position_id: str):
        """Disable trailing stop for a position."""
        if position_id in self.active_trailing_stops:
            self.active_trailing_stops[position_id]['enabled'] = False
            logger.info(f"❌ Trailing stop disabled for position {position_id}")

    def remove_trailing_stop(self, position_id: str):
        """Remove trailing stop configuration."""
        if position_id in self.active_trailing_stops:
            del self.active_trailing_stops[position_id]
            logger.info(f"🗑️ Trailing stop removed for position {position_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get trailing stop statistics."""
        return {
            'active_trailing_stops': len([c for c in self.active_trailing_stops.values() if c['enabled']]),
            'total_configurations': len(self.active_trailing_stops),
            'positions': list(self.active_trailing_stops.keys())
        }


# Singleton instance
_trailing_stop_manager: Optional[TrailingStopManager] = None


def get_trailing_stop_manager() -> TrailingStopManager:
    """Get or create TrailingStopManager singleton."""
    global _trailing_stop_manager
    if _trailing_stop_manager is None:
        _trailing_stop_manager = TrailingStopManager()
        logger.info("✅ TrailingStopManager initialized")
    return _trailing_stop_manager
