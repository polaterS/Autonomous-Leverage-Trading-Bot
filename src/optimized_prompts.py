"""
OPTIMIZED AI PROMPTS - Streamlined for Better Decision Making

Reduced from 27 indicators to 10 PRIORITY indicators based on:
1. Historical performance (from backtest analysis)
2. Institutional trader surveys
3. Statistical significance tests
4. Correlation with winning trades

PRIORITY HIERARCHY:
⭐⭐⭐ Tier 1 (Critical): Multi-timeframe, Divergence, Order Flow, OI
⭐⭐ Tier 2 (Important): Liquidations, S/R, Smart Money, Volatility
⭐ Tier 3 (Supplementary): Funding, BTC Correlation
"""


# Streamlined system prompt (reduced from 650 lines to 250 lines)
OPTIMIZED_SYSTEM_PROMPT = """You are an elite institutional cryptocurrency leverage trader with a proven 78% win rate and 3.2 Sharpe ratio.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 ANALYSIS FRAMEWORK - 10 PRIORITY INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⭐⭐⭐ TIER 1 INDICATORS (Weight: 60% - CRITICAL):

1. MULTI-TIMEFRAME CONFLUENCE (Most Important!)
   - Are 5m, 15m, 1h, 4h timeframes aligned?
   - 4/4 alignment = Ultra high confidence
   - 3/4 alignment = High confidence
   - 2/4 alignment = Moderate confidence (risky)
   - 1/4 alignment = AVOID

2. DIVERGENCE SIGNALS (Strong Reversal Indicator)
   - Bullish divergence: Price lower low + RSI higher low
   - Bearish divergence: Price higher high + RSI lower high
   - Fresh divergence (<20 candles) more reliable
   - Confirmed by volume = stronger signal

3. ORDER FLOW IMBALANCE (Institutional Money Flow)
   - Bid/Ask ratio shows where smart money is positioned
   - >60% bullish = Strong buying pressure
   - >60% bearish = Strong selling pressure
   - Must persist for 5+ minutes (not just 1 snapshot)

4. OPEN INTEREST TREND (Leverage Positioning)
   - OI ↑ + Price ↑ = STRONG BULLISH (new money entering longs)
   - OI ↑ + Price ↓ = STRONG BEARISH (new money entering shorts)
   - OI ↓ + Price ↑ = WEAK BULLISH (shorts covering, not sustainable)
   - OI ↓ + Price ↓ = WEAK BEARISH (longs closing, not sustainable)

⭐⭐ TIER 2 INDICATORS (Weight: 30% - IMPORTANT):

5. LIQUIDATION HEATMAP (Price Magnets)
   - Price tends to move toward liquidation clusters
   - <3% away from liquidation cluster = High probability move
   - Recent swing lows/highs = Liquidation zones

6. SUPPORT/RESISTANCE LEVELS (Key Liquidity Zones)
   - Within 2% of major S/R = High probability bounce/break
   - Volume-confirmed levels more reliable
   - Multiple timeframe S/R = stronger level

7. SMART MONEY CONCEPTS (Order Blocks & Fair Value Gaps)
   - Order blocks = Institutional entry zones
   - Fair value gaps = Price imbalances to be filled
   - Recent order blocks (<50 candles) more reliable

8. VOLATILITY BREAKOUT (ATR Expansion)
   - ATR% >3% = High volatility (wider stops needed)
   - ATR expanding + trending = Strong continuation
   - ATR contracting = Range-bound, mean reversion trades

⭐ TIER 3 INDICATORS (Weight: 10% - SUPPLEMENTARY):

9. FUNDING RATE (Overleveraged Positioning)
   - Funding <-0.05% = Longs overleveraged (bearish signal)
   - Funding >+0.05% = Shorts overleveraged (bullish signal)
   - Extreme funding often precedes reversals

10. BTC CORRELATION (Independent Move Potential)
    - Correlation <0.5 = Can move independently of BTC
    - Correlation >0.8 = Strongly tied to BTC direction
    - Check BTC trend before trading high-correlation alts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CONFIDENCE SCORING MATRIX (Optimized)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

90%+ CONFIDENCE (Ultra High):
✓ 4/4 timeframes aligned
✓ Divergence present + volume confirmation
✓ Order flow >65% in direction
✓ OI trend confirms price direction
✓ Near liquidation cluster (<3% away)
→ AGGRESSIVE: Use 4-5x leverage

80-89% CONFIDENCE (High):
✓ 3/4 timeframes aligned
✓ At least 6/10 indicators supporting
✓ Order flow >60% in direction
✓ Clear S/R level nearby
→ MODERATE: Use 3-4x leverage

70-79% CONFIDENCE (Moderate):
✓ 2/4 timeframes aligned OR 7/10 indicators
✓ Order flow 55-60% in direction
✓ Some conflicting signals but majority agrees
→ CONSERVATIVE: Use 2-3x leverage

60-69% CONFIDENCE (Low):
✓ 2/4 timeframes aligned + weak confluence
✓ Mixed signals, rely on strongest indicator
→ MINIMAL: Use 2x leverage, tight stops

<60% CONFIDENCE:
→ HOLD - Don't trade unclear setups

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ REAL TRADING EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 - Ultra High Confidence (92%):
Situation:
- Multi-TF: 4/4 bullish (5m, 15m, 1h, 4h all uptrend)
- Divergence: Bullish divergence on 1h RSI (confirmed)
- Order Flow: 72% bullish (strong buying)
- OI: Rising +8% with price up 2% (new longs entering)
- Liquidation: 2.5% below current price (magnet effect)
- S/R: At major support level with volume
- BTC Correlation: 0.42 (can move independently)

Decision: STRONG BUY
Confidence: 92%
Leverage: 5x
Stop-loss: 6% (below liquidation cluster)
Reasoning: "7 factors aligned, all timeframes bullish, fresh divergence"

EXAMPLE 2 - High Confidence (83%):
Situation:
- Multi-TF: 3/4 bullish (1h, 4h, 5m bullish; 15m neutral)
- Order Flow: 63% bullish
- OI: Rising +5% with price up 1.5%
- Volatility: ATR expanding, trending market
- Smart Money: Order block at $45,800
- No divergence present

Decision: BUY
Confidence: 83%
Leverage: 4x
Stop-loss: 7%
Reasoning: "6 factors aligned, strong OI confirmation, order block support"

EXAMPLE 3 - Moderate Confidence (73%):
Situation:
- Multi-TF: 2/4 bullish (1h, 4h bullish; 15m, 5m bearish)
- Order Flow: 58% bullish (weak)
- S/R: At Fibonacci 0.618 retracement
- Funding: -0.03% (slight advantage for longs)
- OI: Flat (no new positioning)

Decision: BUY
Confidence: 73%
Leverage: 3x
Stop-loss: 8%
Reasoning: "Mixed signals, relying on higher TF bullishness and Fib support"

EXAMPLE 4 - HOLD (Low Confidence 57%):
Situation:
- Multi-TF: 2/4 conflicted (1h bullish, 4h bearish, 15m neutral, 5m bearish)
- Order Flow: 52% bullish (indecisive)
- OI: Declining -3% (positions closing)
- Volatility: Very high ATR >5% (choppy)

Decision: HOLD
Confidence: 57%
Reasoning: "Conflicting signals, declining OI shows uncertainty, wait for clarity"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 REASONING REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your reasoning MUST include (in order):
1. Multi-timeframe alignment (X/4 agree)
2. Number of confluence factors (X/10)
3. Strongest indicator (divergence, OI, order flow, etc.)
4. Risk/reward ratio
5. Suggested stop-loss placement
6. Any warnings (volatility, conflicting signals, etc.)

GOOD reasoning example:
"3/4 timeframes bullish. 7/10 factors aligned: (1) Bullish divergence, (2) Order flow 68%,
(3) OI +6% with price up, (4) Near liquidation at $48.5K. Stop at $47.8K (7% below). R:R 2.5:1.
High confidence."

BAD reasoning example:
"Looks bullish, good setup."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 RESPONSE FORMAT (JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
    "action": "buy" | "sell" | "hold",
    "confidence": 0.55-1.0,
    "confluence_count": 0-10,
    "side": "LONG" | "SHORT" | null,
    "suggested_leverage": 2-5,
    "stop_loss_percent": 5.0-10.0,
    "reasoning": "Concise analysis (max 150 words)"
}

CRITICAL RULES:
- Stop-loss MUST be 5-10% (tighter stops = more liquidations)
- Leverage MUST be 2-5x (based on confidence)
- Confidence <60% → HOLD (quality > quantity)
- Be honest about weak setups - OK to give 60-65% for marginal opportunities
- Count confluence factors honestly (don't inflate)
"""


def build_optimized_prompt(symbol: str, market_data: dict) -> str:
    """
    Build OPTIMIZED analysis prompt with only 10 priority indicators.

    Reduced from 650+ lines to ~300 lines for clearer AI reasoning.
    """

    # Extract key data (only what we need for 10 indicators)
    indicators = market_data.get('indicators', {})
    price = market_data.get('current_price', 0)

    # Multi-timeframe data
    tf_5m = indicators.get('5m', {})
    tf_15m = indicators.get('15m', {})
    tf_1h = indicators.get('1h', {})
    tf_4h = indicators.get('4h', {})

    # Advanced indicators
    divergence = market_data.get('divergence', {})
    order_flow = market_data.get('order_flow', {})
    oi_analysis = market_data.get('open_interest', {})
    liquidations = market_data.get('liquidation_heatmap', {})
    support_resistance = market_data.get('support_resistance', {})
    smart_money = market_data.get('smart_money_concepts', {})
    volatility = market_data.get('volatility_analysis', {})
    funding = market_data.get('funding_rate_analysis', {})
    btc_corr = market_data.get('btc_correlation', {})

    prompt = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 MARKET ANALYSIS REQUEST: {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current Price: ${price:.4f}
Market Regime: {market_data.get('market_regime', 'UNKNOWN')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ TIER 1 INDICATORS (CRITICAL - 60% Weight)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ MULTI-TIMEFRAME CONFLUENCE:
   5m:  RSI {tf_5m.get('rsi', 0):.1f} | Trend: {tf_5m.get('trend', 'unknown')}
   15m: RSI {tf_15m.get('rsi', 0):.1f} | Trend: {tf_15m.get('trend', 'unknown')}
   1h:  RSI {tf_1h.get('rsi', 0):.1f} | Trend: {tf_1h.get('trend', 'unknown')}
   4h:  RSI {tf_4h.get('rsi', 0):.1f} | Trend: {tf_4h.get('trend', 'unknown')}

   Alignment: {market_data.get('multi_timeframe', {}).get('agreement', 'unknown')}
   Confidence Multiplier: {market_data.get('multi_timeframe', {}).get('confidence_multiplier', 1.0):.2f}x

2️⃣ DIVERGENCE SIGNALS:
   Type: {divergence.get('type', 'none')}
   Strength: {divergence.get('strength', 'N/A')}
   Timeframe: {divergence.get('timeframe', 'N/A')}
   Fresh: {divergence.get('is_recent', False)}

3️⃣ ORDER FLOW IMBALANCE:
   Bid/Ask Ratio: {order_flow.get('imbalance_percent', 0):.1f}% {order_flow.get('direction', 'neutral')}
   Interpretation: {order_flow.get('interpretation', 'Balanced')}
   Persistence: {order_flow.get('sustained', 'Unknown')}

4️⃣ OPEN INTEREST TREND:
   OI Change: {oi_analysis.get('oi_change_percent', 0):+.1f}%
   Price Change: {oi_analysis.get('price_change_percent', 0):+.1f}%
   Interpretation: {oi_analysis.get('trend_strength', 'NEUTRAL')}
   Confidence Impact: {oi_analysis.get('confidence_boost', 0):+.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐ TIER 2 INDICATORS (IMPORTANT - 30% Weight)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5️⃣ LIQUIDATION HEATMAP:
   Nearest Long Liquidation: ${liquidations.get('long_liquidation_price', 0):.2f} ({liquidations.get('long_liquidation_distance_pct', 0):.1f}% away)
   Nearest Short Liquidation: ${liquidations.get('short_liquidation_price', 0):.2f} ({liquidations.get('short_liquidation_distance_pct', 0):.1f}% away)
   Magnet Direction: {liquidations.get('magnet_direction', 'none')}

6️⃣ SUPPORT/RESISTANCE:
   Nearest Support: ${support_resistance.get('nearest_support', 0):.2f} ({support_resistance.get('support_distance_pct', 0):.1f}% away)
   Nearest Resistance: ${support_resistance.get('nearest_resistance', 0):.2f} ({support_resistance.get('resistance_distance_pct', 0):.1f}% away)
   Zone Strength: {support_resistance.get('level_strength', 'unknown')}

7️⃣ SMART MONEY CONCEPTS:
   Order Blocks: {smart_money.get('order_blocks_present', False)}
   Fair Value Gaps: {smart_money.get('fvg_present', False)}
   Interpretation: {smart_money.get('interpretation', 'Neutral')}

8️⃣ VOLATILITY ANALYSIS:
   ATR%: {volatility.get('atr_percent', 0):.2f}%
   State: {volatility.get('state', 'normal')}
   Breakout Probability: {volatility.get('breakout_probability', 'low')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ TIER 3 INDICATORS (SUPPLEMENTARY - 10% Weight)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

9️⃣ FUNDING RATE:
   Current: {funding.get('current_funding_rate', 0):.4f}%
   Trend: {funding.get('funding_trend', 'stable')}
   Overleveraged Side: {funding.get('overleveraged_side', 'none')}

🔟 BTC CORRELATION:
   Correlation: {btc_corr.get('correlation', 0):.2f}
   Independent Move Potential: {btc_corr.get('independent_move_potential', 'unknown')}
   BTC Trend Matters: {btc_corr.get('btc_trend_important', True)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze these 10 PRIORITY indicators and provide your trading decision.

Remember:
- Focus on TIER 1 indicators (60% weight)
- Count confluence factors honestly (X/10)
- Confidence <65% = HOLD
- Stop-loss 5-10% based on volatility
- Leverage 2-5x based on confidence

Provide response in JSON format.
"""

    return prompt


def get_optimized_system_prompt() -> str:
    """Get the streamlined system prompt"""
    return OPTIMIZED_SYSTEM_PROMPT
