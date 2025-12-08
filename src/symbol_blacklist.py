"""
🚫 Symbol Blacklist Manager

Manages list of symbols that should be skipped during scanning due to:
- Exchange restrictions (Binance "Invalid symbol status" errors)
- Repeated failures (3+ consecutive losses)
- Low liquidity / manipulation concerns
- Other technical issues

PROFIT FIX #4: Prevents wasted execution attempts on problematic symbols.
"""

from typing import Set, Dict, Optional
from datetime import datetime, timezone
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SymbolBlacklist:
    """
    Manages symbol blacklist for trading bot.

    Features:
    - Permanent blacklist (exchange restrictions)
    - Temporary blacklist (recent failures, auto-expires)
    - Auto-blacklist on repeated failures
    - Persistence to disk
    """

    def __init__(self, blacklist_file: str = "data/symbol_blacklist.json"):
        self.blacklist_file = Path(blacklist_file)

        # 🚫 PERMANENT BLACKLIST: Exchange restrictions + Low liquidity coins
        self.permanent_blacklist: Set[str] = {
            # Exchange restrictions (Binance won't allow positions)
            'FTM/USDT:USDT',  # Binance: Invalid symbol status
            'MKR/USDT:USDT',  # Binance: Invalid symbol status

            # ═══════════════════════════════════════════════════════════════
            # 🚫 LOW LIQUIDITY / LOW PRICE COINS - Unreliable candle patterns
            # Added: 2025-12-08 (User request after HOT loss)
            # ═══════════════════════════════════════════════════════════════

            # User-identified problematic coins
            'HOT/USDT:USDT',   # Holo - $0.0005, gürültülü mumlar, -$5.41 kayıp
            'SLP/USDT:USDT',   # Smooth Love Potion - $0.0009, düşük fiyat
            'ONT/USDT:USDT',   # Ontology - düşük likidite, belirsiz mumlar
            'FLOW/USDT:USDT',  # Flow - düşük likidite

            # Similar low-price coins (< $0.01)
            'SHIB/USDT:USDT',  # Shiba Inu - meme coin, manipülasyon riski
            'PEPE/USDT:USDT',  # Pepe - meme coin, çok volatil
            'FLOKI/USDT:USDT', # Floki - meme coin
            'LUNC/USDT:USDT',  # Luna Classic - çok düşük fiyat
            'BTTC/USDT:USDT',  # BitTorrent - çok düşük fiyat
            'WIN/USDT:USDT',   # WINkLink - çok düşük fiyat
            'NFT/USDT:USDT',   # APENFT - çok düşük fiyat
            'BONK/USDT:USDT',  # Bonk - meme coin
            '1000SATS/USDT:USDT',  # 1000SATS - düşük fiyat
            'DOGS/USDT:USDT',  # Dogs - meme coin
            'NOT/USDT:USDT',   # Notcoin - meme coin
            'NEIRO/USDT:USDT', # Neiro - meme coin

            # Low liquidity / manipulation prone
            'SPELL/USDT:USDT', # Spell - düşük likidite
            'REEF/USDT:USDT',  # Reef - düşük likidite
            'DENT/USDT:USDT',  # Dent - düşük fiyat, daha önce sorun yaşandı
            'SC/USDT:USDT',    # Siacoin - düşük fiyat
            'ANKR/USDT:USDT',  # Ankr - düşük fiyat
            'PEOPLE/USDT:USDT', # People - düşük likidite
            'JASMY/USDT:USDT', # Jasmy - düşük fiyat
            'ACH/USDT:USDT',   # Alchemy Pay - düşük likidite
            'CTXC/USDT:USDT',  # Cortex - düşük likidite
            'KEY/USDT:USDT',   # SelfKey - düşük likidite
            'STMX/USDT:USDT',  # StormX - düşük likidite
            'OGN/USDT:USDT',   # Origin - düşük likidite
            'CELR/USDT:USDT',  # Celer - düşük fiyat
            'CKB/USDT:USDT',   # Nervos - düşük fiyat
            'BICO/USDT:USDT',  # Biconomy - düşük likidite

            # User-requested additions (2025-12-08 batch 2)
            'SEI/USDT:USDT',   # Sei - kullanıcı talebi, güvenilmez mumlar
            'ZRX/USDT:USDT',   # 0x - kullanıcı talebi, düşük likidite
            'CELO/USDT:USDT',  # Celo - kullanıcı talebi, daha önce sorun yaşandı
            'C98/USDT:USDT',   # Coin98 - kullanıcı talebi, düşük likidite
            'ONE/USDT:USDT',   # Harmony - kullanıcı talebi, güvenilmez
            'ALICE/USDT:USDT', # MyNeighborAlice - kullanıcı talebi, düşük likidite

            # Very volatile / unreliable
            'LUNA/USDT:USDT',  # Luna 2.0 - güvenilmez
            'UST/USDT:USDT',   # UST - ölü coin
            'USTC/USDT:USDT',  # USTC - ölü coin
        }

        # ⏰ TEMPORARY BLACKLIST: Recent failures (auto-expires after 24 hours)
        # Format: {symbol: {'reason': str, 'until': datetime, 'failure_count': int}}
        self.temporary_blacklist: Dict[str, Dict] = {}

        # 📊 FAILURE TRACKING: Track consecutive failures for auto-blacklist
        # Format: {symbol: {'count': int, 'last_failure': datetime}}
        self.failure_tracking: Dict[str, Dict] = {}

        # Load persisted blacklist
        self._load_blacklist()

        logger.info(
            f"🚫 SymbolBlacklist initialized:\n"
            f"   - Permanent: {len(self.permanent_blacklist)} symbols\n"
            f"   - Temporary: {len(self.temporary_blacklist)} symbols\n"
            f"   - File: {self.blacklist_file}"
        )

    def is_blacklisted(self, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Check if symbol is blacklisted.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')

        Returns:
            (is_blacklisted, reason)
        """
        # Check permanent blacklist
        if symbol in self.permanent_blacklist:
            return True, "Permanent blacklist (exchange restriction)"

        # Check temporary blacklist (remove if expired)
        if symbol in self.temporary_blacklist:
            temp_data = self.temporary_blacklist[symbol]
            expiry = temp_data.get('until')

            # Check if expired
            if expiry and datetime.now(timezone.utc) > expiry:
                logger.info(f"   ⏰ Temporary blacklist expired for {symbol}")
                del self.temporary_blacklist[symbol]
                self._save_blacklist()
                return False, None

            return True, temp_data.get('reason', 'Temporary blacklist')

        return False, None

    def add_permanent(self, symbol: str, reason: str = "Manual blacklist"):
        """Add symbol to permanent blacklist."""
        self.permanent_blacklist.add(symbol)
        logger.warning(f"🚫 Added {symbol} to PERMANENT blacklist: {reason}")
        self._save_blacklist()

    def add_temporary(
        self,
        symbol: str,
        reason: str,
        hours: int = 24,
        failure_count: int = 1
    ):
        """
        Add symbol to temporary blacklist.

        Args:
            symbol: Trading symbol
            reason: Reason for blacklist
            hours: Hours until expiry (default 24)
            failure_count: Number of failures that triggered this
        """
        expiry = datetime.now(timezone.utc).timestamp() + (hours * 3600)

        self.temporary_blacklist[symbol] = {
            'reason': reason,
            'until': expiry,
            'failure_count': failure_count,
            'added_at': datetime.now(timezone.utc).isoformat()
        }

        logger.warning(
            f"⏰ Added {symbol} to TEMPORARY blacklist ({hours}h):\n"
            f"   Reason: {reason}\n"
            f"   Failures: {failure_count}\n"
            f"   Expires: {datetime.fromtimestamp(expiry, timezone.utc).isoformat()}"
        )
        self._save_blacklist()

    def track_failure(self, symbol: str, reason: str = "Trade failure"):
        """
        Track failed trade for auto-blacklist.

        Auto-blacklists symbol after 3 consecutive failures.

        Args:
            symbol: Trading symbol
            reason: Failure reason
        """
        now = datetime.now(timezone.utc)

        if symbol not in self.failure_tracking:
            self.failure_tracking[symbol] = {'count': 0, 'last_failure': now}

        self.failure_tracking[symbol]['count'] += 1
        self.failure_tracking[symbol]['last_failure'] = now

        count = self.failure_tracking[symbol]['count']

        logger.warning(f"📉 Failure tracked for {symbol}: {count} consecutive failures")

        # Auto-blacklist after 3 consecutive failures
        if count >= 3:
            self.add_temporary(
                symbol,
                f"Auto-blacklist: {count} consecutive failures - {reason}",
                hours=48,  # 2-day cooldown
                failure_count=count
            )
            # Reset counter
            self.failure_tracking[symbol]['count'] = 0

    def track_success(self, symbol: str):
        """Reset failure counter on successful trade."""
        if symbol in self.failure_tracking:
            logger.info(f"✅ Success! Resetting failure counter for {symbol}")
            del self.failure_tracking[symbol]

    def remove_from_temporary(self, symbol: str):
        """Remove symbol from temporary blacklist."""
        if symbol in self.temporary_blacklist:
            del self.temporary_blacklist[symbol]
            logger.info(f"✅ Removed {symbol} from temporary blacklist")
            self._save_blacklist()

    def get_blacklist_summary(self) -> Dict:
        """Get summary of current blacklist."""
        return {
            'permanent': list(self.permanent_blacklist),
            'temporary': {
                symbol: {
                    'reason': data['reason'],
                    'expires_at': datetime.fromtimestamp(data['until'], timezone.utc).isoformat(),
                    'failure_count': data.get('failure_count', 0)
                }
                for symbol, data in self.temporary_blacklist.items()
            },
            'tracking_failures': {
                symbol: {
                    'consecutive_failures': data['count'],
                    'last_failure': data['last_failure'].isoformat()
                }
                for symbol, data in self.failure_tracking.items()
            }
        }

    def _load_blacklist(self):
        """Load blacklist from disk."""
        try:
            if self.blacklist_file.exists():
                with open(self.blacklist_file, 'r') as f:
                    data = json.load(f)

                # Load permanent blacklist
                if 'permanent' in data:
                    self.permanent_blacklist.update(data['permanent'])

                # Load temporary blacklist (skip expired)
                if 'temporary' in data:
                    now = datetime.now(timezone.utc).timestamp()
                    for symbol, temp_data in data['temporary'].items():
                        if temp_data.get('until', 0) > now:
                            self.temporary_blacklist[symbol] = temp_data

                logger.info(f"✅ Loaded blacklist from {self.blacklist_file}")
        except Exception as e:
            logger.warning(f"Failed to load blacklist: {e}")

    def _save_blacklist(self):
        """Save blacklist to disk."""
        try:
            # Ensure directory exists
            self.blacklist_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'permanent': list(self.permanent_blacklist),
                'temporary': self.temporary_blacklist,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

            with open(self.blacklist_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"💾 Saved blacklist to {self.blacklist_file}")
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")


# Singleton instance
_symbol_blacklist: Optional[SymbolBlacklist] = None


def get_symbol_blacklist() -> SymbolBlacklist:
    """Get or create SymbolBlacklist singleton."""
    global _symbol_blacklist
    if _symbol_blacklist is None:
        _symbol_blacklist = SymbolBlacklist()
    return _symbol_blacklist
