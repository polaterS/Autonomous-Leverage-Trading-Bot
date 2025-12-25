"""
Utility functions and logging setup for the trading bot.
"""

import logging
import colorlog
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any
import json
import time


class TurkeyTimeFormatter(colorlog.ColoredFormatter):
    """Custom formatter that uses Turkey Time (UTC+3)."""

    def formatTime(self, record, datefmt=None):
        """Override formatTime to use Turkey Time (UTC+3)."""
        # Convert timestamp to Turkey Time (UTC+3)
        dt = datetime.fromtimestamp(record.created)
        turkey_offset = timedelta(hours=3)
        turkey_time = dt + turkey_offset

        if datefmt:
            return turkey_time.strftime(datefmt)
        else:
            return turkey_time.strftime('%Y-%m-%d %H:%M:%S')


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up colored logging with appropriate level."""

    log_level = logging.DEBUG if debug else logging.INFO

    # Create logger
    logger = logging.getLogger('trading_bot')

    # If logger already has handlers, don't add more (prevent duplicates)
    if logger.handlers:
        return logger

    # Create color formatter with Turkey Time
    formatter = TurkeyTimeFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Set up handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.setLevel(log_level)
    logger.addHandler(handler)

    # Suppress noisy libraries
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    return logger


def calculate_liquidation_price(
    entry_price: Decimal,
    leverage: int,
    side: str,
    maintenance_margin_rate: Decimal = Decimal("0.004")
) -> Decimal:
    """
    Calculate liquidation price for a leveraged position.

    Args:
        entry_price: Entry price of the position
        leverage: Leverage multiplier
        side: 'LONG' or 'SHORT'
        maintenance_margin_rate: Exchange maintenance margin rate (default 0.4%)

    Returns:
        Liquidation price
    """
    # Simplified liquidation calculation (Binance-like)
    # Actual formula: Liq Price = Entry Price × (1 ± (1 - Maintenance Margin Rate) / Leverage)

    leverage_decimal = Decimal(str(leverage))

    if side == 'LONG':
        # For longs, liquidation is below entry
        liq_price = entry_price * (1 - (1 - maintenance_margin_rate) / leverage_decimal)
    else:  # SHORT
        # For shorts, liquidation is above entry
        liq_price = entry_price * (1 + (1 - maintenance_margin_rate) / leverage_decimal)

    return liq_price


def calculate_position_size(
    capital: Decimal,
    position_size_percent: Decimal,
    entry_price: Decimal,
    leverage: int,
    fixed_position_usd: Optional[Decimal] = None
) -> tuple[Decimal, Decimal]:
    """
    Calculate position quantity and value.

    Args:
        capital: Current capital
        position_size_percent: Position size as % of capital (used if fixed_position_usd is None)
        entry_price: Entry price
        leverage: Leverage multiplier
        fixed_position_usd: If set, uses this fixed USD amount instead of percentage

    Returns:
        (quantity, position_value_usd)
    """
    # FIXED USD SIZING: Use fixed amount if specified (e.g., always $100)
    if fixed_position_usd is not None:
        position_value = fixed_position_usd
        logger = logging.getLogger('trading_bot')
        logger.info(f"💰 Using FIXED position size: ${float(position_value):.2f} (not {float(position_size_percent*100):.1f}% of capital)")
    else:
        # PERCENTAGE SIZING: Calculate position value as % of capital
        position_value = capital * position_size_percent

    # Calculate quantity from position value
    # 🔴 IMPORTANT: position_value should ALREADY include leverage!
    # The caller (trade_executor.py) multiplies margin × leverage before passing it here.
    #
    # Example with 20x leverage:
    # - Margin: $41 (capital allocated for this position)
    # - Leverage: 20x
    # - Position value (passed in): $41 × 20 = $820
    # - Entry price: $0.52
    # - Quantity: $820 / $0.52 = 1,577 coins
    # - This opens a $820 position using $41 margin
    #
    # ⚠️ DON'T multiply by leverage here - it's already in position_value!
    quantity = position_value / entry_price

    return quantity, position_value


def calculate_stop_loss_price(
    entry_price: Decimal,
    stop_loss_percent: Decimal,
    side: str,
    leverage: Optional[int] = None,
    position_value: Optional[Decimal] = None
) -> Decimal:
    """
    Calculate stop-loss price based on maximum USD loss (accounting for leverage).

    CRITICAL FIX: The stop-loss must account for leverage!
    - Without leverage consideration: 10% price stop = 10% loss
    - With 10x leverage: 10% price stop = 100% loss (LIQUIDATION!)

    The correct formula:
    - Max USD loss = position_value * stop_loss_percent / 100
    - Price stop % = (Max USD loss / position_value) / leverage

    Args:
        entry_price: Entry price
        stop_loss_percent: Maximum loss as % of position value (e.g., 10 for 10%)
        side: 'LONG' or 'SHORT'
        leverage: Position leverage (if None, uses old simple calculation)
        position_value: Position value in USD (if None, uses old simple calculation)

    Returns:
        Stop-loss price
    """
    # NEW: Leverage-adjusted stop-loss calculation
    if leverage is not None and position_value is not None and leverage > 1:
        # Calculate price move % that results in the desired USD loss
        # Example: $100 position, 10% max loss = $10, 9x leverage
        # Price move % = $10 / $100 / 9 = 1.11% (not 10%!)
        max_usd_loss_percent = stop_loss_percent / 100  # e.g., 0.02 for 2%
        price_move_percent = max_usd_loss_percent / Decimal(str(leverage))

        # 🔥🔥🔥 CRITICAL FIX v4 (2025-12): Balanced stop-loss for ~$12 max loss
        #
        # HISTORY:
        # - v2: 2% = $20 loss (too wide, big losses)
        # - v3: 0.8% = $8 loss (too tight, positions closed before developing!)
        #
        # USER FEEDBACK:
        # - Before $6 stop-loss: 85% win rate (11/13 trades)
        # - After $6 stop-loss: 0% win rate (0/6 trades)
        # - Positions need ROOM TO BREATHE!
        #
        # SOLUTION (v4): 1.2% price move = ~$12 max loss
        # - Gives position time to develop
        # - Still much better than $20 losses
        # - Combined with TREND + ADX filters for quality entries
        #
        # With different leverage:
        # - 10x leverage, $100 margin: 1.2% move = $12 loss
        # - 15x leverage, $66 margin: 1.2% move = $12 loss
        # - 20x leverage, $50 margin: 1.2% move = $12 loss
        MIN_PRICE_MOVE_PERCENT = Decimal("0.012")  # 1.2% = ~$12 max loss on $1000 position

        logger = logging.getLogger('trading_bot')
        logger.info(f"🛡️ SL FIX v4 ACTIVE: Min price move = {float(MIN_PRICE_MOVE_PERCENT)*100:.2f}% (target: ~$12 max loss)")

        if price_move_percent < MIN_PRICE_MOVE_PERCENT:
            logger.warning(
                f"⚠️ SL TOO TIGHT! Calculated {float(price_move_percent)*100:.3f}% → "
                f"Enforcing minimum {float(MIN_PRICE_MOVE_PERCENT)*100:.1f}%"
            )
            price_move_percent = MIN_PRICE_MOVE_PERCENT

        if side == 'LONG':
            # For longs, stop-loss is below entry
            sl_price = entry_price * (1 - price_move_percent)
        else:  # SHORT
            # For shorts, stop-loss is above entry
            sl_price = entry_price * (1 + price_move_percent)

        logger = logging.getLogger('trading_bot')
        logger.info(
            f"🎯 LEVERAGE-ADJUSTED STOP-LOSS: {side} {leverage}x | "
            f"Max loss: {float(stop_loss_percent):.1f}% (${float(position_value * max_usd_loss_percent):.2f}) | "
            f"Price stop: {float(price_move_percent * 100):.2f}% | "
            f"Entry: ${float(entry_price):.4f} → Stop: ${float(sl_price):.4f}"
        )

        return sl_price

    # OLD: Simple percentage-based calculation (fallback for legacy code)
    sl_decimal = stop_loss_percent / 100

    if side == 'LONG':
        # For longs, stop-loss is below entry
        sl_price = entry_price * (1 - sl_decimal)
    else:  # SHORT
        # For shorts, stop-loss is above entry
        sl_price = entry_price * (1 + sl_decimal)

    return sl_price


def calculate_min_profit_price(
    entry_price: Decimal,
    min_profit_usd: Decimal,
    position_value: Decimal,
    leverage: int,
    side: str
) -> Decimal:
    """Calculate the price needed to achieve minimum profit target."""

    # Calculate required price change percentage
    # profit = position_value * price_change_pct * leverage
    # price_change_pct = profit / (position_value * leverage)
    leverage_decimal = Decimal(str(leverage))
    required_price_change_pct = min_profit_usd / (position_value * leverage_decimal)

    if side == 'LONG':
        # For longs, profit comes from price increase
        target_price = entry_price * (1 + required_price_change_pct)
    else:  # SHORT
        # For shorts, profit comes from price decrease
        target_price = entry_price * (1 - required_price_change_pct)

    return target_price


def calculate_pnl(
    entry_price: Decimal,
    current_price: Decimal,
    quantity: Decimal,
    side: str,
    leverage: int,
    position_value: Decimal,
    include_fees: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive P&L metrics WITH trading fees.

    Args:
        entry_price: Position entry price
        current_price: Current market price
        quantity: Position quantity
        side: 'LONG' or 'SHORT'
        leverage: Position leverage
        position_value: Position value in USD
        include_fees: Include Binance trading fees (default: True)

    Returns dict with:
        - unrealized_pnl: NET dollar amount (after fees)
        - gross_pnl: GROSS dollar amount (before fees)
        - total_fees: Entry + estimated exit fees
        - pnl_percent: Percentage of position value
        - leveraged_pnl_percent: Percentage including leverage effect
        - price_change_pct: Raw price change
    """

    # Price change percentage
    if side == 'LONG':
        price_change_pct = (current_price - entry_price) / entry_price
    else:  # SHORT
        price_change_pct = (entry_price - current_price) / entry_price

    # P&L Calculation (GROSS - before fees)
    # 🔥 CRITICAL FIX: position_value ALREADY includes leverage!
    # position_value = margin × leverage (e.g., $40 × 25 = $1000)
    # So we should NOT multiply by leverage again!
    leverage_decimal = Decimal(str(leverage))
    gross_pnl = position_value * price_change_pct  # Don't multiply by leverage!

    # Calculate trading fees (Binance futures taker: 0.05%)
    total_fees = Decimal("0")
    if include_fees:
        taker_fee_rate = Decimal("0.0005")  # 0.05%

        # Entry fee
        entry_notional = quantity * entry_price
        entry_fee = entry_notional * taker_fee_rate

        # Exit fee (estimated at current price)
        exit_notional = quantity * current_price
        exit_fee = exit_notional * taker_fee_rate

        total_fees = entry_fee + exit_fee

    # NET P&L (after fees)
    unrealized_pnl = gross_pnl - total_fees

    return {
        'unrealized_pnl': unrealized_pnl,  # NET (after fees)
        'gross_pnl': gross_pnl,  # GROSS (before fees)
        'total_fees': total_fees,
        'pnl_percent': float((unrealized_pnl / position_value) * 100),
        'leveraged_pnl_percent': float(price_change_pct * leverage_decimal * 100),
        'price_change_pct': float(price_change_pct)
    }


def calculate_profit_targets(
    entry_price: Decimal,
    side: str,
    position_value: Decimal,
    leverage: int,
    stop_loss_price: Decimal,
    market_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Decimal]:
    """
    Calculate DYNAMIC profit targets based on S/R levels and market conditions.

    🎯 v6.5 DYNAMIC TP SYSTEM:
    - Uses actual S/R levels from PA analysis as targets
    - Falls back to R/R-based targets if no S/R data
    - Adjusts targets based on volatility (ATR)
    - Ensures minimum R/R ratio for profitability

    PROFESSIONAL APPROACH:
    - LONG: TP at nearest resistance levels
    - SHORT: TP at nearest support levels
    - Dynamic min_profit based on position size and volatility

    Args:
        entry_price: Entry price of position
        side: 'LONG' or 'SHORT'
        position_value: Position value in USD
        leverage: Leverage used
        stop_loss_price: Actual stop-loss price (leverage-adjusted)
        market_data: Market data with S/R levels, ATR, and PA analysis

    Returns:
        Dict with profit_target_1, profit_target_2, target_1_profit_usd, target_2_profit_usd,
        dynamic_min_profit_usd, target_source
    """
    logger = logging.getLogger('trading_bot')
    leverage_decimal = Decimal(str(leverage))
    
    # 🔥 Calculate risk distance (entry to stop-loss)
    risk_distance = abs(entry_price - stop_loss_price)
    risk_percent = float(risk_distance / entry_price * 100)
    
    logger.info(
        f"💰 Calculating DYNAMIC profit targets:\n"
        f"   Entry: ${float(entry_price):.4f}\n"
        f"   Stop: ${float(stop_loss_price):.4f}\n"
        f"   Risk Distance: ${float(risk_distance):.4f} ({risk_percent:.2f}%)"
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 STEP 1: Try to use S/R levels from PA analysis (PROFESSIONAL APPROACH)
    # ═══════════════════════════════════════════════════════════════════════════
    target_source = "R/R-based"  # Default
    profit_target_1 = None
    profit_target_2 = None
    
    if market_data:
        pa_analysis = market_data.get('pa_analysis', {})
        
        # Get S/R levels from PA analysis
        resistance_levels = []
        support_levels = []
        
        # Try different data formats
        if pa_analysis:
            # Format 1: Direct from should_enter_trade result
            if 'targets' in pa_analysis:
                targets = pa_analysis['targets']
                if side == 'LONG' and len(targets) >= 2:
                    profit_target_1 = Decimal(str(targets[0]))
                    profit_target_2 = Decimal(str(targets[1]))
                    target_source = "PA-targets"
                elif side == 'SHORT' and len(targets) >= 2:
                    profit_target_1 = Decimal(str(targets[0]))
                    profit_target_2 = Decimal(str(targets[1]))
                    target_source = "PA-targets"
            
            # Format 2: S/R levels from analysis
            if not profit_target_1:
                sr_data = pa_analysis.get('sr_analysis', {})
                resistance_levels = sr_data.get('resistance', [])
                support_levels = sr_data.get('support', [])
        
        # Format 3: Direct S/R from market_data
        if not profit_target_1:
            if 'resistance_levels' in market_data:
                resistance_levels = market_data.get('resistance_levels', [])
            if 'support_levels' in market_data:
                support_levels = market_data.get('support_levels', [])
        
        # 🎯 LONG: Target resistance levels
        if side == 'LONG' and resistance_levels and not profit_target_1:
            # Sort resistance levels by price (ascending)
            if isinstance(resistance_levels[0], dict):
                sorted_res = sorted([r['price'] for r in resistance_levels if r['price'] > float(entry_price)])
            else:
                sorted_res = sorted([r for r in resistance_levels if r > float(entry_price)])
            
            if len(sorted_res) >= 2:
                profit_target_1 = Decimal(str(sorted_res[0]))  # First resistance
                profit_target_2 = Decimal(str(sorted_res[1]))  # Second resistance
                target_source = "S/R-levels"
                logger.info(f"   🎯 Using RESISTANCE levels as targets: T1=${float(profit_target_1):.4f}, T2=${float(profit_target_2):.4f}")
            elif len(sorted_res) == 1:
                profit_target_1 = Decimal(str(sorted_res[0]))
                # T2 = T1 + risk distance (extend beyond first resistance)
                profit_target_2 = profit_target_1 + risk_distance
                target_source = "S/R-extended"
        
        # 🎯 SHORT: Target support levels
        elif side == 'SHORT' and support_levels and not profit_target_1:
            # Sort support levels by price (descending - closest first)
            if isinstance(support_levels[0], dict):
                sorted_sup = sorted([s['price'] for s in support_levels if s['price'] < float(entry_price)], reverse=True)
            else:
                sorted_sup = sorted([s for s in support_levels if s < float(entry_price)], reverse=True)
            
            if len(sorted_sup) >= 2:
                profit_target_1 = Decimal(str(sorted_sup[0]))  # First support
                profit_target_2 = Decimal(str(sorted_sup[1]))  # Second support
                target_source = "S/R-levels"
                logger.info(f"   🎯 Using SUPPORT levels as targets: T1=${float(profit_target_1):.4f}, T2=${float(profit_target_2):.4f}")
            elif len(sorted_sup) == 1:
                profit_target_1 = Decimal(str(sorted_sup[0]))
                # T2 = T1 - risk distance (extend beyond first support)
                profit_target_2 = profit_target_1 - risk_distance
                target_source = "S/R-extended"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 STEP 2: Fallback to R/R-based targets if no S/R data
    # ═══════════════════════════════════════════════════════════════════════════
    if not profit_target_1 or not profit_target_2:
        logger.info("   ⚠️ No S/R levels found, using R/R-based targets")
        target_source = "R/R-based"
        
        if side == 'LONG':
            # TARGET 1: 1.5x risk distance
            reward_distance_1 = risk_distance * Decimal("1.5")
            profit_target_1 = entry_price + reward_distance_1
            
            # TARGET 2: 2.5x risk distance (more aggressive for dynamic system)
            reward_distance_2 = risk_distance * Decimal("2.5")
            profit_target_2 = entry_price + reward_distance_2
        else:  # SHORT
            reward_distance_1 = risk_distance * Decimal("1.5")
            profit_target_1 = entry_price - reward_distance_1
            
            reward_distance_2 = risk_distance * Decimal("2.5")
            profit_target_2 = entry_price - reward_distance_2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 STEP 3: Validate targets meet minimum R/R requirements
    # ═══════════════════════════════════════════════════════════════════════════
    MIN_RR_T1 = Decimal("1.2")  # Minimum 1.2:1 for T1
    MIN_RR_T2 = Decimal("2.0")  # Minimum 2.0:1 for T2
    
    # Calculate actual R/R for each target
    t1_reward = abs(profit_target_1 - entry_price)
    t2_reward = abs(profit_target_2 - entry_price)
    
    t1_rr = t1_reward / risk_distance if risk_distance > 0 else Decimal("0")
    t2_rr = t2_reward / risk_distance if risk_distance > 0 else Decimal("0")
    
    # Adjust T1 if R/R too low
    if t1_rr < MIN_RR_T1:
        logger.info(f"   ⚠️ T1 R/R {float(t1_rr):.2f} < {float(MIN_RR_T1)}, adjusting...")
        if side == 'LONG':
            profit_target_1 = entry_price + (risk_distance * MIN_RR_T1)
        else:
            profit_target_1 = entry_price - (risk_distance * MIN_RR_T1)
        t1_rr = MIN_RR_T1
    
    # Adjust T2 if R/R too low
    if t2_rr < MIN_RR_T2:
        logger.info(f"   ⚠️ T2 R/R {float(t2_rr):.2f} < {float(MIN_RR_T2)}, adjusting...")
        if side == 'LONG':
            profit_target_2 = entry_price + (risk_distance * MIN_RR_T2)
        else:
            profit_target_2 = entry_price - (risk_distance * MIN_RR_T2)
        t2_rr = MIN_RR_T2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 STEP 4: Calculate DYNAMIC minimum profit based on conditions
    # ═══════════════════════════════════════════════════════════════════════════
    # Base minimum profit (from config, typically $5)
    from src.config import get_settings
    settings = get_settings()
    base_min_profit = settings.min_profit_usd
    
    # Get ATR for volatility adjustment
    atr_percent = Decimal("2.5")  # Default medium volatility
    if market_data:
        indicators = market_data.get('indicators', {})
        if isinstance(indicators, dict):
            atr_percent = Decimal(str(indicators.get('atr_percent', 2.5)))
    
    # 🎯 DYNAMIC MIN PROFIT CALCULATION
    # Higher volatility = higher potential profit = higher min target
    # Larger position = higher absolute profit potential
    
    # Volatility multiplier (1.0 - 2.0 based on ATR)
    if atr_percent < Decimal("1.5"):
        volatility_mult = Decimal("0.8")  # Low vol = lower target
    elif atr_percent < Decimal("3.0"):
        volatility_mult = Decimal("1.0")  # Medium vol = base target
    elif atr_percent < Decimal("5.0"):
        volatility_mult = Decimal("1.5")  # High vol = higher target
    else:
        volatility_mult = Decimal("2.0")  # Very high vol = much higher target
    
    # Position size multiplier (larger positions can target more)
    # $100 position = 1.0x, $500 position = 1.5x, $1000+ = 2.0x
    if position_value < Decimal("100"):
        size_mult = Decimal("0.8")
    elif position_value < Decimal("300"):
        size_mult = Decimal("1.0")
    elif position_value < Decimal("500"):
        size_mult = Decimal("1.3")
    elif position_value < Decimal("1000"):
        size_mult = Decimal("1.6")
    else:
        size_mult = Decimal("2.0")
    
    # Calculate dynamic min profit
    dynamic_min_profit = base_min_profit * volatility_mult * size_mult
    
    # Cap at reasonable levels
    dynamic_min_profit = max(dynamic_min_profit, Decimal("3.0"))  # Minimum $3
    dynamic_min_profit = min(dynamic_min_profit, Decimal("50.0"))  # Maximum $50
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 STEP 5: Calculate expected profits
    # ═══════════════════════════════════════════════════════════════════════════
    # Target 1: 50% of position
    price_change_1 = abs(profit_target_1 - entry_price) / entry_price
    target_1_profit_usd = (position_value * Decimal("0.5")) * price_change_1 * leverage_decimal
    
    # Target 2: Remaining 50% (total profit if both hit)
    price_change_2 = abs(profit_target_2 - entry_price) / entry_price
    target_2_total_profit_usd = (
        target_1_profit_usd +
        (position_value * Decimal("0.5")) * price_change_2 * leverage_decimal
    )
    
    logger.info(
        f"✅ DYNAMIC Profit targets calculated ({target_source}):\n"
        f"   Target 1: ${float(profit_target_1):.4f} (R/R {float(t1_rr):.1f}:1) → ${float(target_1_profit_usd):.2f} profit\n"
        f"   Target 2: ${float(profit_target_2):.4f} (R/R {float(t2_rr):.1f}:1) → ${float(target_2_total_profit_usd):.2f} total\n"
        f"   Dynamic Min Profit: ${float(dynamic_min_profit):.2f} (base ${float(base_min_profit):.2f} × vol {float(volatility_mult):.1f} × size {float(size_mult):.1f})\n"
        f"   ATR: {float(atr_percent):.2f}% | Position: ${float(position_value):.2f}"
    )
    
    return {
        'profit_target_1': profit_target_1,
        'profit_target_2': profit_target_2,
        'target_1_profit_usd': target_1_profit_usd,
        'target_2_profit_usd': target_2_total_profit_usd,
        'dynamic_min_profit_usd': dynamic_min_profit,
        'target_source': target_source,
        't1_rr_ratio': float(t1_rr),
        't2_rr_ratio': float(t2_rr),
    }


def format_duration(seconds: int) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def parse_ai_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse AI response JSON with error handling.

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            return None

        json_str = response_text[start_idx:end_idx + 1]
        return json.loads(json_str)

    except Exception as e:
        logging.getLogger('trading_bot').error(f"Failed to parse AI response: {e}")
        return None


def is_bullish(indicators: Dict[str, float]) -> bool:
    """
    Determine if technical indicators are bullish.

    Checks:
    - RSI between 40-70 (not overbought)
    - MACD above signal line
    - Price above SMA
    """
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    price = indicators.get('close', 0)
    sma = indicators.get('sma_20', price)

    conditions = [
        40 < rsi < 70,  # Not oversold or overbought
        macd > macd_signal,  # MACD bullish
        price > sma  # Price above SMA
    ]

    # Require at least 2 out of 3 conditions
    return sum(conditions) >= 2


def is_bearish(indicators: Dict[str, float]) -> bool:
    """
    Determine if technical indicators are bearish.

    Checks:
    - RSI between 30-60 (not oversold)
    - MACD below signal line
    - Price below SMA
    """
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    price = indicators.get('close', 0)
    sma = indicators.get('sma_20', price)

    conditions = [
        30 < rsi < 60,  # Not oversold or overbought
        macd < macd_signal,  # MACD bearish
        price < sma  # Price below SMA
    ]

    # Require at least 2 out of 3 conditions
    return sum(conditions) >= 2


def safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert value to Decimal."""
    try:
        if value is None:
            return default
        return Decimal(str(value))
    except:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except:
        return default
