"""
Risk Management System for the trading bot.
Enforces all safety rules and validates trades before execution.
"""

from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import date
from src.config import get_settings
from src.database import get_db_client
from src.utils import setup_logging, calculate_liquidation_price

logger = setup_logging()


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self):
        self.settings = get_settings()

    async def validate_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive trade validation against all risk rules.

        Args:
            trade_params: Dict with symbol, side, leverage, stop_loss_percent, current_price
                         market_breadth (optional): Dict with bullish_percent, bearish_percent, neutral_percent

        Returns:
            Dict with 'approved': bool, 'reason': str, 'adjusted_params': dict
        """
        symbol = trade_params['symbol']
        side = trade_params['side']
        leverage = trade_params['leverage']
        stop_loss_percent = trade_params['stop_loss_percent']
        current_price = Decimal(str(trade_params['current_price']))

        logger.info(f"Validating trade: {symbol} {side} {leverage}x with {stop_loss_percent}% stop-loss")

        # 🔥 CRITICAL FIX #1: MARKET DIRECTION FILTER
        # Prevent opening trades against clear market direction
        market_breadth = trade_params.get('market_breadth')
        if market_breadth:
            bullish_pct = market_breadth.get('bullish_percent', 0)
            bearish_pct = market_breadth.get('bearish_percent', 0)
            neutral_pct = market_breadth.get('neutral_percent', 0)

            # Reject SHORT trades if market not sufficiently bearish
            if side == 'SHORT':
                # 🔥 USER REQUESTED: Reduced threshold to 3% (ultra permissive)
                # Only blocks SHORT when market is EXTREMELY bullish (97%+ bullish)
                if bearish_pct < 3:  # Market not bearish enough
                    return {
                        'approved': False,
                        'reason': f'Market not bearish enough for SHORT (Bearish: {bearish_pct:.0f}%). Need ≥3% bearish.',
                        'adjusted_params': None
                    }
                logger.info(f"✓ Market direction OK for SHORT: {bearish_pct:.0f}% bearish (≥3% threshold)")

            # Reject LONG trades if market not sufficiently bullish
            elif side == 'LONG':
                # 🔥 USER REQUESTED: Reduced threshold to 3% (ultra permissive)
                # Only blocks LONG when market is EXTREMELY bearish (97%+ bearish)
                if bullish_pct < 3:  # Market not bullish enough
                    return {
                        'approved': False,
                        'reason': f'Market not bullish enough for LONG (Bullish: {bullish_pct:.0f}%). Need ≥3% bullish.',
                        'adjusted_params': None
                    }
                logger.info(f"✓ Market direction OK for LONG: {bullish_pct:.0f}% bullish (≥3% threshold)")

        # 🎯 CRITICAL PRE-TRADE TECHNICAL VALIDATION
        # These checks prevent 30-50% of losing trades!
        tech_valid = await self._validate_technical_setup(trade_params)
        if not tech_valid['approved']:
            return tech_valid

        # RULE 1: Stop-loss must be within configured range (reads from database)
        db = await get_db_client()
        config = await db.get_trading_config()

        min_sl = float(config['min_stop_loss_percent'])  # Already in percentage format (e.g., 4.0 = 4%)
        max_sl = float(config['max_stop_loss_percent'])  # Already in percentage format (e.g., 5.0 = 5%)

        # 🔧 DISABLED: Stop-loss validation (user wants ±$1 limits only)
        # Stop-loss is set to 50% as emergency safety net
        # if stop_loss_percent < min_sl or stop_loss_percent > max_sl:
        #     return {
        #         'approved': False,
        #         'reason': f'Stop-loss {stop_loss_percent}% outside required range ({min_sl}-{max_sl}%)',
        #         'adjusted_params': None
        #     }

        # RULE 2: Check if we have enough capital
        current_capital = await db.get_current_capital()
        position_value = current_capital * self.settings.position_size_percent

        if position_value < 10:  # Minimum $10 position
            return {
                'approved': False,
                'reason': f'Insufficient capital: ${current_capital:.2f} (need at least $12.50)',
                'adjusted_params': None
            }

        # RULE 3: Leverage check - ENFORCE 6x-10x RANGE
        min_leverage = self.settings.min_leverage  # 6x
        max_leverage = self.settings.max_leverage  # 10x

        if leverage > max_leverage:
            # Adjust leverage down to max (10x)
            logger.warning(f"Leverage {leverage}x exceeds maximum, adjusting to {max_leverage}x")
            leverage = max_leverage
            trade_params['leverage'] = leverage

        if leverage < min_leverage:
            # Adjust leverage up to minimum (6x)
            logger.warning(f"Leverage {leverage}x below minimum, adjusting to {min_leverage}x")
            leverage = min_leverage
            trade_params['leverage'] = min_leverage

        # RULE 4: Check concurrent positions limit (FIXED AT 5)
        # Conservative mode: Maximum 5 positions regardless of capital
        # Each position = $50 (5% of capital)
        active_positions = await db.get_active_positions()
        current_capital = await db.get_current_capital()

        # Fixed maximum: 5 positions (conservative mode)
        max_concurrent = self.settings.max_concurrent_positions

        if len(active_positions) >= max_concurrent:
            return {
                'approved': False,
                'reason': f'Maximum concurrent positions reached ({len(active_positions)}/{max_concurrent}). Capital: ${float(current_capital):.2f}',
                'adjusted_params': None
            }

        # Check if same symbol already has an open position
        for pos in active_positions:
            if pos['symbol'] == trade_params['symbol']:
                return {
                    'approved': False,
                    'reason': f'Position already open for {trade_params["symbol"]}. Cannot open duplicate.',
                    'adjusted_params': None
                }

        # 🎯 #6: RULE 4B: CROSS-SYMBOL CORRELATION CHECK
        # Prevent over-concentration in highly correlated assets
        try:
            from src.ml_pattern_learner import get_ml_learner
            ml_learner = await get_ml_learner()

            correlation_check = ml_learner.check_correlation_risk(
                trade_params['symbol'],
                active_positions
            )

            if not correlation_check['safe']:
                logger.warning(
                    f"🔗 Correlation risk detected for {trade_params['symbol']}: "
                    f"{correlation_check['reason']}"
                )
                return {
                    'approved': False,
                    'reason': f"Correlation risk: {correlation_check['reason']}",
                    'adjusted_params': None,
                    'correlation_details': correlation_check
                }
            else:
                logger.info(
                    f"✅ Correlation check passed: {correlation_check['reason']} "
                    f"(avg: {correlation_check['avg_correlation']:.1%})"
                )
        except Exception as e:
            # Non-critical: allow trade if correlation check fails
            logger.warning(f"Correlation check failed (allowing trade): {e}")

        # RULE 5: Daily loss limit check - DISABLED
        # User wants per-trade $10 loss limit (handled in stop-loss), not daily limit
        # Daily loss limit is now disabled to allow multiple trades per day
        # Each trade has its own $10 max loss via stop-loss calculation

        # RULE 6: Consecutive losses check - DISABLED BY USER REQUEST
        # REASON: User wants bot to keep trading regardless of loss streak
        # Protection now relies on:
        # - Per-trade loss limits: -$1.50 to -$2.50
        # - Stop-loss: 1.5-2.5%
        # - Maximum 10 concurrent positions (diversification)
        #
        # OLD: Stopped trading after 5-6 consecutive losses
        # NEW: Never stop trading due to loss streak
        consecutive_losses = await db.get_consecutive_losses()
        logger.debug(
            f"📊 Consecutive losses: {consecutive_losses} "
            f"(check disabled - bot will continue trading)"
        )

        # RULE 7: Maximum loss per trade check
        max_loss_per_trade = position_value * (Decimal(str(stop_loss_percent)) / 100) * leverage
        max_acceptable_loss = current_capital * Decimal("0.20")  # 20% of capital max

        if max_loss_per_trade > max_acceptable_loss:
            # Try to reduce leverage (but stay within 6x-10x range)
            adjusted_leverage = int((max_acceptable_loss / position_value) / (Decimal(str(stop_loss_percent)) / 100))
            adjusted_leverage = max(self.settings.min_leverage, min(adjusted_leverage, self.settings.max_leverage))

            if adjusted_leverage < self.settings.min_leverage:
                return {
                    'approved': False,
                    'reason': f'Potential loss ${max_loss_per_trade:.2f} exceeds 20% of capital even with minimum leverage ({self.settings.min_leverage}x)',
                    'adjusted_params': None
                }

            logger.warning(f"Adjusting leverage from {leverage}x to {adjusted_leverage}x to limit risk")
            leverage = adjusted_leverage
            trade_params['leverage'] = adjusted_leverage

        # RULE 8: Liquidation distance - DISABLED BY USER REQUEST
        # REASON: 15-20x leverage requires tight liquidation distance (~6-7%)
        # Protection now relies on:
        # - Tight stop-loss (1.5-2.5%)
        # - Fixed loss limit (-$1.50 to -$2.50)
        # - Position monitoring every 60 seconds
        #
        # OLD: Required 8% liquidation distance, blocked all 15-20x trades
        # NEW: Trust stop-loss system instead of liquidation distance
        liq_price = calculate_liquidation_price(current_price, leverage, side)
        liq_distance = abs(current_price - liq_price) / current_price

        logger.debug(
            f"📊 Liquidation distance: {float(liq_distance)*100:.1f}% "
            f"(liq price: ${float(liq_price):.4f}, leverage: {leverage}x, "
            f"check disabled - trust stop-loss)"
        )

        # RULE 9: Minimum profit target validation
        min_profit_pct = self.settings.min_profit_usd / position_value

        if min_profit_pct > Decimal("0.15"):  # More than 15% needed
            return {
                'approved': False,
                'reason': f'Minimum profit target {float(min_profit_pct)*100:.1f}% too high for position size',
                'adjusted_params': None
            }

        # RULE 10: Trading enabled check
        config = await db.get_trading_config()
        if not config.get('is_trading_enabled', True):
            return {
                'approved': False,
                'reason': 'Trading is currently disabled',
                'adjusted_params': None
            }

        # All checks passed
        logger.info(f"✅ Trade validation passed for {symbol} {side} {leverage}x")

        return {
            'approved': True,
            'reason': 'All risk checks passed',
            'adjusted_params': trade_params
        }

    async def check_position_risk(self, position: Dict[str, Any], current_price: Decimal) -> Dict[str, Any]:
        """
        Check risk metrics for an open position.

        Returns:
            Dict with risk assessment and recommended actions
        """
        symbol = position['symbol']
        side = position['side']
        entry_price = Decimal(str(position['entry_price']))
        leverage = position['leverage']
        stop_loss_price = Decimal(str(position['stop_loss_price']))
        liquidation_price = Decimal(str(position['liquidation_price']))

        # Calculate distance to liquidation
        liq_distance = abs(current_price - liquidation_price) / current_price

        # Calculate distance to stop-loss
        sl_distance = abs(current_price - stop_loss_price) / current_price

        # Determine risk level
        if liq_distance < Decimal("0.05"):  # Less than 5%
            risk_level = 'CRITICAL'
            action = 'CLOSE_IMMEDIATELY'
            reason = f'Liquidation risk: only {float(liq_distance)*100:.2f}% away'

        elif liq_distance < Decimal("0.10"):  # Less than 10%
            risk_level = 'HIGH'
            action = 'MONITOR_CLOSELY'
            reason = f'Liquidation warning: {float(liq_distance)*100:.2f}% away'

        elif sl_distance < Decimal("0.01"):  # Stop-loss very close
            risk_level = 'HIGH'
            action = 'STOP_LOSS_IMMINENT'
            reason = f'Stop-loss about to trigger: {float(sl_distance)*100:.2f}% away'

        else:
            risk_level = 'NORMAL'
            action = 'CONTINUE'
            reason = 'Risk levels acceptable'

        return {
            'risk_level': risk_level,
            'action': action,
            'reason': reason,
            'liq_distance_pct': float(liq_distance) * 100,
            'sl_distance_pct': float(sl_distance) * 100
        }

    async def can_open_position(self) -> Dict[str, Any]:
        """
        Quick check if we can open a new position (before full validation).

        Checks:
        1. Sufficient capital
        2. Not at max concurrent positions
        3. Trading enabled (circuit breakers, etc.)

        Returns:
            Dict with 'can_open': bool, 'reason': str
        """
        db = await get_db_client()

        # Check 1: Sufficient capital
        current_capital = await db.get_current_capital()
        min_required_capital = 10 / self.settings.position_size_percent  # Minimum for $10 position

        if current_capital < min_required_capital:
            return {
                'can_open': False,
                'reason': f'Insufficient capital: ${float(current_capital):.2f} < ${float(min_required_capital):.2f} required'
            }

        # Check 2: Max concurrent positions (FIXED AT 5)
        active_positions = await db.get_active_positions()

        # Fixed maximum: 5 positions (conservative mode)
        max_concurrent = self.settings.max_concurrent_positions

        if len(active_positions) >= max_concurrent:
            return {
                'can_open': False,
                'reason': f'Maximum concurrent positions reached ({len(active_positions)}/{max_concurrent})'
            }

        # Check 3: Daily limits (circuit breakers, etc.)
        daily_limits = await self.check_daily_limits()
        if not daily_limits['can_trade']:
            return {
                'can_open': False,
                'reason': daily_limits['reason']
            }

        return {
            'can_open': True,
            'reason': f'Ready to trade ({len(active_positions)}/{max_concurrent} positions, ${float(current_capital):.2f} capital)'
        }

    async def check_daily_limits(self) -> Dict[str, Any]:
        """
        Check if daily trading limits have been reached.

        Includes:
        - Maximum drawdown protection (20%)
        - Consecutive losses
        - Manual disable

        Returns:
            Dict with 'can_trade': bool, 'reason': str
        """
        db = await get_db_client()

        # 🚨 CRITICAL: MAX DRAWDOWN PROTECTION (20%)
        # Prevents catastrophic losses by stopping trading if capital drops too much
        config = await db.get_trading_config()
        current_capital = Decimal(str(config['current_capital']))
        starting_capital = Decimal(str(config.get('starting_capital', current_capital)))

        # Calculate drawdown from session/day start
        if starting_capital > 0:
            drawdown_percent = float(((starting_capital - current_capital) / starting_capital) * 100)

            # Maximum drawdown allowed (20% = stop trading if daily loss exceeds 20%)
            # 🔴 LIVE TRADING: Back to normal 20% threshold
            # Database starting capital fixed to $100, so drawdown calculation is accurate
            max_allowed_drawdown = 20.0

            # Only trigger if we're in LOSS (drawdown > 0)
            # If drawdown < 0, we're in PROFIT (ignore)
            if drawdown_percent >= max_allowed_drawdown and drawdown_percent > 0:
                logger.critical(
                    f"🚨 MAX DRAWDOWN REACHED: {drawdown_percent:.1f}% "
                    f"(${float(starting_capital):.2f} → ${float(current_capital):.2f})"
                )

                # Auto-disable trading
                await db.set_trading_enabled(False)

                # Send critical alert
                from src.telegram_notifier import get_notifier
                notifier = get_notifier()
                await notifier.send_alert(
                    'critical',
                    f"🚨 MAX DRAWDOWN PROTECTION ACTIVATED\n\n"
                    f"Drawdown: {drawdown_percent:.1f}% (Max: {max_allowed_drawdown:.0f}%)\n"
                    f"Starting Capital: ${float(starting_capital):.2f}\n"
                    f"Current Capital: ${float(current_capital):.2f}\n"
                    f"Loss: ${float(starting_capital - current_capital):.2f}\n\n"
                    f"🛑 Trading STOPPED automatically for safety\n"
                    f"Review performance before restarting with /startbot"
                )

                return {
                    'can_trade': False,
                    'reason': f'Max drawdown reached ({drawdown_percent:.1f}%)',
                    'type': 'max_drawdown'
                }

            # Warning at 15% drawdown (before hitting limit)
            # Only warn if in LOSS (drawdown > 0)
            elif drawdown_percent >= 15.0 and drawdown_percent > 0:
                logger.warning(f"⚠️ High drawdown: {drawdown_percent:.1f}% (approaching 20% limit)")
            elif drawdown_percent < 0:
                # We're in profit! Log positive performance
                profit_percent = abs(drawdown_percent)
                logger.info(f"📈 Daily profit: +{profit_percent:.1f}% (${float(starting_capital):.2f} → ${float(current_capital):.2f})")

        # Check consecutive losses - DISABLED BY USER REQUEST
        # User wants bot to continue trading regardless of loss streak
        consecutive_losses = await db.get_consecutive_losses()
        logger.debug(
            f"📊 Consecutive losses: {consecutive_losses} "
            f"(limit check disabled - bot continues trading)"
        )

        # Check trading enabled
        if not config.get('is_trading_enabled', True):
            return {
                'can_trade': False,
                'reason': 'Trading manually disabled',
                'type': 'manual_disable'
            }

        return {
            'can_trade': True,
            'reason': 'All limits OK',
            'type': None
        }

    async def calculate_optimal_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_percent: float,
        leverage: int,
        ai_confidence: float = 0.75,
        opportunity_score: float = 75.0
    ) -> Decimal:
        """
        Calculate optimal position size using REAL BALANCE from Binance.

        🔥 USER REQUESTED: Use actual exchange balance, not database capital
        Example: 116 USDT balance → 2 positions = 58 USDT per position

        Factors considered:
        1. Real exchange balance (from Binance)
        2. Number of active positions + this new position
        3. AI confidence and opportunity score adjustments
        4. Maximum risk limits

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_percent: Stop-loss percentage
            leverage: Leverage multiplier
            ai_confidence: AI confidence (0-1)
            opportunity_score: Opportunity score (0-100)

        Returns:
            Position value in USD
        """
        db = await get_db_client()

        # 🔥 USE REAL BALANCE FROM EXCHANGE (not database capital)
        from src.exchange_client import get_exchange_client
        exchange = get_exchange_client()  # Singleton, not async
        real_balance = await exchange.fetch_balance()  # This is async

        # Get active positions to calculate how to split balance
        active_positions = await db.get_active_positions()
        current_position_count = len(active_positions)

        # Calculate how many positions we can have (max from settings)
        max_positions = self.settings.max_concurrent_positions  # 2 positions

        # Split balance equally among max positions
        # Example: 116 USDT / 2 positions = 58 USDT per position
        base_size_per_position = real_balance / Decimal(str(max_positions))

        logger.info(
            f"💰 Real balance position sizing: "
            f"Balance: ${float(real_balance):.2f} USDT | "
            f"Max positions: {max_positions} | "
            f"Current positions: {current_position_count} | "
            f"Base size per position: ${float(base_size_per_position):.2f}"
        )

        # METHOD 1: Real balance size (primary method)
        static_size = base_size_per_position

        # METHOD 2: Kelly Criterion with historical win rate (using real balance)
        kelly_size = await self._calculate_kelly_position_size(
            real_balance, stop_loss_percent, leverage
        )

        # METHOD 3: Confidence-adjusted sizing
        # Reduce size if confidence is low, increase if confidence is very high
        confidence_multiplier = Decimal(str(min(max(ai_confidence, 0.5), 1.0)))  # 0.5-1.0 range
        confidence_adjusted_size = static_size * confidence_multiplier

        # METHOD 4: Opportunity score adjustment
        # Score 65-80: reduce size by 10%
        # Score 80-90: full size
        # Score 90+: increase size by 10%
        if opportunity_score < 80:
            score_multiplier = Decimal("0.90")  # Reduce by 10%
        elif opportunity_score >= 90:
            score_multiplier = Decimal("1.10")  # Increase by 10%
        else:
            score_multiplier = Decimal("1.0")   # Full size

        opportunity_adjusted_size = confidence_adjusted_size * score_multiplier

        # Choose the MOST CONSERVATIVE of all methods
        optimal_size = min(
            static_size,
            kelly_size,
            opportunity_adjusted_size
        )

        # Absolute maximum risk check (never risk more than 15% per trade, using real balance)
        max_risk = real_balance * Decimal("0.15")
        stop_loss_decimal = Decimal(str(stop_loss_percent)) / 100
        max_position_from_risk = max_risk / (stop_loss_decimal * leverage)

        optimal_size = min(optimal_size, max_position_from_risk)

        # Ensure minimum viable position
        if optimal_size < 10:
            logger.warning(f"Calculated position size ${optimal_size:.2f} too small")
            return Decimal("0")

        logger.info(
            f"Dynamic Position Sizing - "
            f"Static: ${static_size:.2f}, "
            f"Kelly: ${kelly_size:.2f}, "
            f"Confidence Adj: ${confidence_adjusted_size:.2f}, "
            f"Opportunity Adj: ${opportunity_adjusted_size:.2f} → "
            f"Final: ${optimal_size:.2f}"
        )

        return optimal_size

    async def _calculate_kelly_position_size(
        self,
        capital: Decimal,
        stop_loss_percent: float,
        leverage: int
    ) -> Decimal:
        """
        Calculate position size using Kelly Criterion.

        Kelly % = W - (1 - W) / R
        Where:
        - W = Win rate (probability of winning)
        - R = Win/Loss ratio (avg win / avg loss)

        Args:
            capital: Current capital
            stop_loss_percent: Stop-loss percentage
            leverage: Leverage multiplier

        Returns:
            Kelly-optimal position size
        """
        db = await get_db_client()

        # Get recent trade history (last 20 trades for statistical significance)
        recent_trades = await db.get_recent_trades(limit=20)

        if len(recent_trades) < 10:
            # Not enough history - use conservative 50% of static size
            logger.debug("Kelly Criterion: Insufficient trade history, using conservative sizing")
            return capital * self.settings.position_size_percent * Decimal("0.5")

        # Calculate win rate
        winning_trades = [t for t in recent_trades if t['is_winner']]
        win_rate = len(winning_trades) / len(recent_trades)

        # Calculate average win and average loss
        avg_win = (
            sum(Decimal(str(t['realized_pnl_usd'])) for t in winning_trades) / len(winning_trades)
            if winning_trades else Decimal("0")
        )

        losing_trades = [t for t in recent_trades if not t['is_winner']]
        avg_loss = (
            abs(sum(Decimal(str(t['realized_pnl_usd'])) for t in losing_trades) / len(losing_trades))
            if losing_trades else Decimal("1")  # Prevent division by zero
        )

        # Calculate win/loss ratio
        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = Decimal("2.0")  # Default assumption

        # Kelly Criterion formula: K = W - (1 - W) / R
        kelly_percent = Decimal(str(win_rate)) - (Decimal("1") - Decimal(str(win_rate))) / win_loss_ratio

        # Use fractional Kelly (50% of Kelly) for safety
        fractional_kelly = kelly_percent * Decimal("0.5")

        # Clamp Kelly between 10% and 100% of capital
        kelly_percent_clamped = max(Decimal("0.10"), min(fractional_kelly, Decimal("1.0")))

        kelly_position_size = capital * kelly_percent_clamped

        logger.debug(
            f"Kelly Criterion: Win Rate={win_rate:.1%}, "
            f"W/L Ratio={float(win_loss_ratio):.2f}, "
            f"Kelly%={float(kelly_percent)*100:.1f}%, "
            f"Fractional={float(fractional_kelly)*100:.1f}%, "
            f"Size=${kelly_position_size:.2f}"
        )

        return kelly_position_size

    async def should_emergency_close(self, position: Dict[str, Any], current_price: Decimal) -> tuple[bool, str]:
        """
        Determine if position should be emergency closed.

        USER REQUEST: Only use Binance's real unrealized PNL, close at exactly ±$1

        Returns:
            (should_close: bool, reason: str)
        """
        # ====================================================================
        # 🚫 LIQUIDATION PANIC: DISABLED
        # ====================================================================
        # USER COMPLAINT: "çok hızlı kapatıyor pozisyonu daha binance tarafında 1 dolar kar olmadan kapatıyor"
        # Bot was closing positions at 3-4% liquidation distance which is SAFE with 25x leverage
        #
        # OLD CODE (DISABLED):
        # liquidation_price = Decimal(str(position['liquidation_price']))
        # liq_distance = abs(current_price - liquidation_price) / current_price
        # if liq_distance < Decimal("0.05"):
        #     return True, f"EMERGENCY: Liquidation risk ({float(liq_distance)*100:.2f}% away)"
        # ====================================================================

        # ====================================================================
        # ❌ STOP-LOSS CHECK: DISABLED (User wants ±$1 targets only)
        # ====================================================================
        # USER REQUEST: "1 dolar kar etti mi hemen kapancak. aynı şekilde 1 dolar zarar etti mi hemen kapanacak"
        # Stop-loss set to 50% as wide safety net - ±$1 limits control exits
        #
        # OLD CODE (DISABLED):
        # from src.utils import calculate_pnl
        # pnl_data = calculate_pnl(...)
        # unrealized_pnl = pnl_data['unrealized_pnl']
        # MAX_LOSS_USD = Decimal("-10.00")
        # if unrealized_pnl <= MAX_LOSS_USD:
        #     return True, f"Stop-loss triggered: ${float(unrealized_pnl):.2f} loss"
        # ====================================================================

        # NO EMERGENCY CLOSE CONDITIONS
        # Let position_monitor.py handle ±$1 profit/loss targets using Binance PNL
        return False, ""

    # 🎯 #9: DYNAMIC STOP-LOSS WITH VOLATILITY

    def calculate_dynamic_stop_loss_distance(
        self,
        atr_percent: float,
        leverage: int
    ) -> float:
        """
        🎯 #9: Calculate optimal stop-loss distance based on ATR percentile.

        Low volatility = tighter stops (5%)
        Normal volatility = medium stops (6-7%)
        High volatility = wider stops (8-10%)

        Args:
            atr_percent: ATR as percentage of price
            leverage: Position leverage (affects stop distance)

        Returns:
            Stop-loss distance as percentage (e.g., 7.0 for 7%)
        """
        # ATR thresholds based on typical crypto volatility
        LOW_VOL_THRESHOLD = 1.5  # < 1.5% ATR = low volatility
        NORMAL_VOL_THRESHOLD = 3.0  # < 3.0% ATR = normal volatility
        # > 3.0% ATR = high volatility

        # Base stop-loss distances
        if atr_percent < LOW_VOL_THRESHOLD:
            # Low volatility: Tighter stop
            base_stop = 5.0
            logger.debug(f"📊 Low volatility (ATR {atr_percent:.2f}%) → Tight stop: {base_stop}%")
        elif atr_percent < NORMAL_VOL_THRESHOLD:
            # Normal volatility: Medium stop
            # Linear interpolation between 5% and 8%
            base_stop = 5.0 + (atr_percent - LOW_VOL_THRESHOLD) * (3.0 / (NORMAL_VOL_THRESHOLD - LOW_VOL_THRESHOLD))
            logger.debug(f"📊 Normal volatility (ATR {atr_percent:.2f}%) → Medium stop: {base_stop:.1f}%")
        else:
            # High volatility: Wider stop
            # Cap at 10% for safety
            base_stop = min(8.0 + (atr_percent - NORMAL_VOL_THRESHOLD) * 0.5, 10.0)
            logger.debug(f"📊 High volatility (ATR {atr_percent:.2f}%) → Wide stop: {base_stop:.1f}%")

        # Adjust for leverage (higher leverage = tighter stops for safety)
        if leverage >= 5:
            leverage_adjustment = 0.8  # 20% tighter
        elif leverage >= 4:
            leverage_adjustment = 0.9  # 10% tighter
        else:
            leverage_adjustment = 1.0  # No adjustment

        final_stop = base_stop * leverage_adjustment

        # Clamp to safe range [5%, 10%]
        final_stop = max(5.0, min(10.0, final_stop))

        logger.info(f"🎯 Dynamic stop-loss: {final_stop:.1f}% (ATR: {atr_percent:.2f}%, Leverage: {leverage}x)")

        return final_stop

    async def update_trailing_stop(
        self,
        position: Dict[str, Any],
        current_price: Decimal,
        atr_percent: float
    ) -> Optional[Decimal]:
        """
        🎯 #9: Update trailing stop-loss if price moved favorably.

        Stops trail price at ATR-based distance but never move against you.

        Args:
            position: Active position dict
            current_price: Current market price
            atr_percent: Current ATR as percentage

        Returns:
            New stop-loss price if updated, None if no update needed
        """
        symbol = position['symbol']
        side = position['side']
        entry_price = Decimal(str(position['entry_price']))
        current_stop = Decimal(str(position['stop_loss_price']))
        leverage = position['leverage']

        # Calculate dynamic stop distance based on current volatility
        stop_distance_pct = self.calculate_dynamic_stop_loss_distance(atr_percent, leverage)
        stop_distance = Decimal(str(stop_distance_pct)) / 100

        # Calculate new trailing stop based on current price
        if side == 'LONG':
            new_stop = current_price * (1 - stop_distance)

            # Only update if new stop is HIGHER than current stop (never move down)
            # AND price has moved at least 2% above entry (don't trail too early)
            price_gain = (current_price - entry_price) / entry_price

            if price_gain >= Decimal("0.02") and new_stop > current_stop:
                logger.info(
                    f"📈 Trailing stop UP: {symbol} LONG | "
                    f"Old: ${float(current_stop):.4f} → New: ${float(new_stop):.4f} "
                    f"(+{float((new_stop - current_stop) / current_stop * 100):.2f}%)"
                )
                return new_stop

        elif side == 'SHORT':
            new_stop = current_price * (1 + stop_distance)

            # Only update if new stop is LOWER than current stop (never move up)
            # AND price has moved at least 2% below entry
            price_gain = (entry_price - current_price) / entry_price

            if price_gain >= Decimal("0.02") and new_stop < current_stop:
                logger.info(
                    f"📉 Trailing stop DOWN: {symbol} SHORT | "
                    f"Old: ${float(current_stop):.4f} → New: ${float(new_stop):.4f} "
                    f"(-{float((current_stop - new_stop) / current_stop * 100):.2f}%)"
                )
                return new_stop

        # No update needed
        return None

    async def _validate_technical_setup(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 CRITICAL: Technical validation before opening position.

        Prevents 30-50% of losing trades by checking:
        1. Support/Resistance distance (avoid immediate rejection)
        2. Volume confirmation (ensure institutional participation)
        3. Order flow validation (confirm directional pressure)
        4. Multi-timeframe alignment (prevent counter-trend trades)

        Args:
            trade_params: Trade parameters including market_data

        Returns:
            Dict with 'approved': bool, 'reason': str
        """
        symbol = trade_params['symbol']
        side = trade_params['side']
        current_price = Decimal(str(trade_params['current_price']))

        # Get market data (if provided)
        market_data = trade_params.get('market_data', {})

        if not market_data:
            logger.warning(f"⚠️ No market_data provided for technical validation, allowing trade")
            return {'approved': True, 'reason': 'No market data to validate', 'adjusted_params': trade_params}

        # ========================================================================
        # CHECK 1: SUPPORT/RESISTANCE DISTANCE (CRITICAL!)
        # ========================================================================
        # Prevent entering near major S/R levels that will cause immediate rejection

        indicators = market_data.get('indicators', {})
        sr_levels = indicators.get('15m', {}).get('support_resistance', {})

        if sr_levels:
            if side == 'LONG':
                # Check distance to nearest resistance
                resistance = sr_levels.get('nearest_resistance')
                if resistance:
                    resistance = Decimal(str(resistance))
                    distance_to_resistance = (resistance - current_price) / current_price * 100

                    if distance_to_resistance < 2.0:  # Less than 2% room to resistance
                        logger.warning(
                            f"🚫 LONG rejected: Too close to resistance | "
                            f"Current: ${float(current_price):.4f} | "
                            f"Resistance: ${float(resistance):.4f} | "
                            f"Distance: {float(distance_to_resistance):.2f}% (need ≥2%)"
                        )
                        return {
                            'approved': False,
                            'reason': f'Too close to resistance ${float(resistance):.4f} (only {float(distance_to_resistance):.1f}% room)',
                            'adjusted_params': None
                        }

                    logger.info(f"✅ S/R Check: {float(distance_to_resistance):.1f}% to resistance")

            elif side == 'SHORT':
                # Check distance to nearest support
                support = sr_levels.get('nearest_support')
                if support:
                    support = Decimal(str(support))
                    distance_to_support = (current_price - support) / current_price * 100

                    if distance_to_support < 2.0:  # Less than 2% room to support
                        logger.warning(
                            f"🚫 SHORT rejected: Too close to support | "
                            f"Current: ${float(current_price):.4f} | "
                            f"Support: ${float(support):.4f} | "
                            f"Distance: {float(distance_to_support):.2f}% (need ≥2%)"
                        )
                        return {
                            'approved': False,
                            'reason': f'Too close to support ${float(support):.4f} (only {float(distance_to_support):.1f}% room)',
                            'adjusted_params': None
                        }

                    logger.info(f"✅ S/R Check: {float(distance_to_support):.1f}% to support")

        # ========================================================================
        # CHECK 2: VOLUME CONFIRMATION (HIGH PRIORITY!)
        # ========================================================================
        # Ensure sufficient volume for reliable price movement

        volume_data = indicators.get('15m', {})
        current_volume = volume_data.get('volume')
        volume_sma = volume_data.get('volume_sma')

        if current_volume and volume_sma:
            volume_ratio = current_volume / volume_sma

            if volume_ratio < 1.2:  # Need 20%+ above average
                logger.warning(
                    f"🚫 Trade rejected: Insufficient volume | "
                    f"Current: {float(current_volume):.0f} | "
                    f"Avg: {float(volume_sma):.0f} | "
                    f"Ratio: {volume_ratio:.2f}x (need ≥1.2x)"
                )
                return {
                    'approved': False,
                    'reason': f'Insufficient volume ({volume_ratio:.2f}x average, need ≥1.2x)',
                    'adjusted_params': None
                }

            logger.info(f"✅ Volume Check: {volume_ratio:.2f}x average")

        # ========================================================================
        # CHECK 3: ORDER FLOW VALIDATION (INSTITUTIONAL CONFIRMATION)
        # ========================================================================
        # Ensure directional pressure from large players

        order_flow = market_data.get('order_flow', {})
        imbalance = order_flow.get('weighted_imbalance', 0)

        if side == 'LONG':
            if imbalance < 5.0:  # Need strong buy pressure
                logger.warning(
                    f"🚫 LONG rejected: Weak buy pressure | "
                    f"Order flow imbalance: {imbalance:.1f}% (need ≥5%)"
                )
                return {
                    'approved': False,
                    'reason': f'Weak buy pressure (order flow: {imbalance:.1f}%, need ≥5%)',
                    'adjusted_params': None
                }
            logger.info(f"✅ Order Flow: Strong buy pressure ({imbalance:.1f}%)")

        elif side == 'SHORT':
            if imbalance > -5.0:  # Need strong sell pressure
                logger.warning(
                    f"🚫 SHORT rejected: Weak sell pressure | "
                    f"Order flow imbalance: {imbalance:.1f}% (need ≤-5%)"
                )
                return {
                    'approved': False,
                    'reason': f'Weak sell pressure (order flow: {imbalance:.1f}%, need ≤-5%)',
                    'adjusted_params': None
                }
            logger.info(f"✅ Order Flow: Strong sell pressure ({imbalance:.1f}%)")

        # ========================================================================
        # CHECK 4: MULTI-TIMEFRAME ALIGNMENT (PREVENT COUNTER-TREND)
        # ========================================================================
        # Ensure trade aligns with higher timeframe bias

        mtf_analysis = market_data.get('multi_timeframe', {})
        confluence = mtf_analysis.get('confluence_analysis', {})
        trading_bias = confluence.get('trading_bias', 'NEUTRAL')

        if side == 'LONG' and trading_bias == 'SHORT_PREFERRED':
            logger.warning(
                f"🚫 LONG rejected: Higher timeframes bearish | "
                f"MTF bias: {trading_bias}"
            )
            return {
                'approved': False,
                'reason': f'Counter-trend trade (MTF bias: {trading_bias})',
                'adjusted_params': None
            }

        if side == 'SHORT' and trading_bias == 'LONG_PREFERRED':
            logger.warning(
                f"🚫 SHORT rejected: Higher timeframes bullish | "
                f"MTF bias: {trading_bias}"
            )
            return {
                'approved': False,
                'reason': f'Counter-trend trade (MTF bias: {trading_bias})',
                'adjusted_params': None
            }

        logger.info(f"✅ MTF Check: Bias is {trading_bias}")

        # ========================================================================
        # ALL CHECKS PASSED!
        # ========================================================================
        logger.info(f"🎯 Technical validation PASSED for {symbol} {side}")

        return {
            'approved': True,
            'reason': 'All technical checks passed (S/R, volume, order flow, MTF)',
            'adjusted_params': trade_params
        }


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
