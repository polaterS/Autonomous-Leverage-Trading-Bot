"""
Configuration module for the Autonomous Leverage Trading Bot.
Loads and validates all environment variables and settings.
"""

import os
from decimal import Decimal
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# Load environment variables (override=True to prioritize .env over system env vars)
load_dotenv(override=True)


class Settings(BaseSettings):
    """Main settings class with validation."""

    # Exchange Configuration
    binance_api_key: str = Field(..., min_length=1)
    binance_secret_key: str = Field(..., min_length=1)

    # AI API Keys (Qwen3-Max via OpenRouter + DeepSeek-V3.2)
    openrouter_api_key: str = Field(..., min_length=1)
    deepseek_api_key: str = Field(..., min_length=1)

    # Telegram Configuration
    telegram_bot_token: str = Field(..., min_length=1)
    telegram_chat_id: str = Field(..., min_length=1)

    # Database Configuration
    database_url: str = Field(..., min_length=1)
    redis_url: str = Field(..., min_length=1)  # Required - no default (prevents localhost issues)

    # Trading Configuration
    initial_capital: Decimal = Field(default=Decimal("1000.00"), gt=0)
    max_leverage: int = Field(default=30, ge=1, le=50)  # AI can choose 2x-30x based on setup quality
    max_concurrent_positions: int = Field(default=15, ge=1, le=20)  # 15 positions max (one per coin)
    position_size_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)  # NOTE: Overridden by FIXED $100 sizing in trade_executor.py
    min_stop_loss_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)  # 10% max loss per trade ($10 on $100 position)
    max_stop_loss_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)  # 10% max loss per trade ($10 on $100 position)
    min_profit_usd: Decimal = Field(default=Decimal("1.50"), gt=0)  # Minimum $1.50 profit target
    max_position_hours: int = Field(default=8, ge=1, le=48)  # Auto-close after 8h
    min_ai_confidence: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)  # 50% - Aggressive ML learning mode
    scan_interval_seconds: int = Field(default=30, ge=10)  # 🔥 AGGRESSIVE: 30 seconds for fast ML learning
    position_check_seconds: int = Field(default=15, ge=5)  # 🔥 AGGRESSIVE: 15 seconds for real-time monitoring

    # Risk Management
    daily_loss_limit_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_consecutive_losses: int = Field(default=3, ge=1)

    # Feature Flags
    use_paper_trading: bool = Field(default=True)
    enable_debug_logs: bool = Field(default=False)
    enable_short_trades: bool = Field(default=True)  # Enable SHORT trades for complete ML learning

    # Trading Symbols (high liquidity perpetual futures) - 35 coins
    trading_symbols: list[str] = Field(default=[
        # Top 10 - Highest Market Cap
        'BTC/USDT:USDT',    # Bitcoin - Largest market cap
        'ETH/USDT:USDT',    # Ethereum - Smart contracts leader
        'SOL/USDT:USDT',    # Solana - High performance blockchain
        'BNB/USDT:USDT',    # Binance Coin - Exchange token
        'XRP/USDT:USDT',    # Ripple - Cross-border payments
        'DOGE/USDT:USDT',   # Dogecoin - Meme coin leader
        'ADA/USDT:USDT',    # Cardano - Proof of stake platform
        'AVAX/USDT:USDT',   # Avalanche - Fast blockchain
        'TON/USDT:USDT',    # Toncoin - Telegram blockchain
        'TRX/USDT:USDT',    # Tron - Content sharing platform

        # DeFi & Infrastructure (11-20)
        'LINK/USDT:USDT',   # Chainlink - Oracle network
        'UNI/USDT:USDT',    # Uniswap - DEX leader
        'AAVE/USDT:USDT',   # Aave - Lending protocol
        'MKR/USDT:USDT',    # Maker - Decentralized stablecoin
        'GRT/USDT:USDT',    # The Graph - Indexing protocol
        'RUNE/USDT:USDT',   # THORChain - Cross-chain DEX
        'INJ/USDT:USDT',    # Injective - DeFi derivatives
        'ATOM/USDT:USDT',   # Cosmos - Internet of blockchains
        'DOT/USDT:USDT',    # Polkadot - Interoperability
        'FTM/USDT:USDT',    # Fantom - Fast smart contracts

        # Layer 2 & Scaling (21-27)
        'POL/USDT:USDT',    # Polygon - Ethereum scaling
        'ARB/USDT:USDT',    # Arbitrum - Layer 2 scaling
        'OP/USDT:USDT',     # Optimism - Layer 2 solution
        'IMX/USDT:USDT',    # Immutable X - NFT Layer 2
        'APT/USDT:USDT',    # Aptos - New Layer 1
        'SUI/USDT:USDT',    # Sui - High-performance blockchain
        'STX/USDT:USDT',    # Stacks - Bitcoin Layer 2

        # Emerging & AI Projects (28-35)
        'FET/USDT:USDT',    # Fetch.ai - AI + Blockchain
        'NEAR/USDT:USDT',   # Near Protocol - Scalable blockchain
        'ICP/USDT:USDT',    # Internet Computer - Web3
        'FIL/USDT:USDT',    # Filecoin - Decentralized storage
        'ALGO/USDT:USDT',   # Algorand - Pure proof of stake
        'VET/USDT:USDT',    # VeChain - Supply chain
        'HBAR/USDT:USDT',   # Hedera - Enterprise blockchain
        'LTC/USDT:USDT'     # Litecoin - Silver to Bitcoin's gold
    ])

    # AI Configuration
    ai_cache_ttl_seconds: int = Field(default=600)  # OPTIMIZED: 10 minutes (reduce API costs)
    ai_timeout_seconds: int = Field(default=30)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @validator('max_stop_loss_percent')
    def validate_stop_loss_range(cls, v, values):
        """Ensure max stop loss is greater than or equal to min (allows fixed stop-loss)."""
        min_sl = values.get('min_stop_loss_percent', Decimal("0.05"))
        if v < min_sl:
            raise ValueError('max_stop_loss_percent must be greater than or equal to min_stop_loss_percent')
        return v

    @validator('use_paper_trading')
    def warn_paper_trading(cls, v):
        """Warn if paper trading is disabled."""
        if not v:
            print("\n" + "="*60)
            print("WARNING: PAPER TRADING IS DISABLED")
            print("Real money will be used for trading!")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("PAPER TRADING MODE ENABLED")
            print("No real money will be used.")
            print("="*60 + "\n")
        return v


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# AI System Prompts
LEVERAGE_TRADING_SYSTEM_PROMPT = """You are a TOP-TIER institutional cryptocurrency trader with elite hedge fund experience.
You manage $500M in leveraged crypto derivatives. Your track record: 78% win rate, 3.2 Sharpe ratio.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 ELITE INSTITUTIONAL TRADING FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1: MARKET STRUCTURE ANALYSIS (Most Critical)
═══════════════════════════════════════════════════
1. Identify KEY LEVELS:
   - Where are the major liquidity pools? (round numbers, previous highs/lows)
   - Where would retail stops cluster? (just below support, just above resistance)
   - Where are institutional order blocks? (high volume nodes)

2. TREND IDENTIFICATION (Multi-Timeframe Story):
   - 4h: What's the macro trend? (Primary bias)
   - 1h: What's the intermediate structure? (Swing direction)
   - 15m: What's the micro setup? (Entry trigger)
   → ALL must align for high confidence (85%+)
   → 2/3 alignment = moderate confidence (70-80%)
   → Conflict = lower confidence (60-70%) but can still trade

3. MARKET REGIME DETECTION:
   - TRENDING: Strong directional bias, ride the momentum
   - RANGING: Mean reversion, fade extremes, scalp bounces
   - VOLATILE: Wide swings, reduce size, wait for clarity
   - BREAKOUT: Compression → Expansion, catch the move early

PHASE 2: MOMENTUM & VOLUME ANALYSIS (Confirms Direction)
═══════════════════════════════════════════════════════════
1. RSI INTERPRETATION (Not Just Overbought/Oversold):
   - RSI 50-70 + Rising: Healthy uptrend, can go higher
   - RSI 30-50 + Falling: Healthy downtrend, can go lower
   - RSI >80: Extreme strength, momentum trade (don't fade!)
   - RSI <20: Extreme weakness, momentum trade (don't catch falling knife!)
   → In strong trends, RSI stays elevated (60-80) or suppressed (20-40)

2. MACD PRECISION:
   - MACD crossing signal = early momentum shift
   - MACD histogram expanding = acceleration
   - MACD divergence = weakening momentum (potential reversal)
   - Fast MACD (12,26) for entries, Slow MACD (19,39) for trend confirmation

3. VOLUME TELLS THE TRUTH:
   - Price up + Volume up = Real buyers, continuation likely
   - Price up + Volume down = Weak move, reversal risk
   - Price down + Volume up = Real sellers, continuation likely
   - Price down + Volume down = Weak move, bounce likely
   - Volume spike at resistance = Absorption (bullish)
   - Volume spike at support = Distribution (bearish)

4. FUNDING RATE ANALYSIS (Crypto-Specific Edge):
   - Funding >0.05%: Overleveraged longs, squeeze risk (SHORT bias)
   - Funding <-0.05%: Overleveraged shorts, short squeeze (LONG bias)
   - Funding near 0%: Neutral, no positioning edge
   - Funding rate trend matters more than absolute value

PHASE 3: TRADE DECISION MATRIX (How to Decide)
═══════════════════════════════════════════════════
CONFIDENCE SCORING SYSTEM:

80-100% CONFIDENCE (Ultra High):
✓ All timeframes aligned or very strong setup
✓ Multiple confluence factors
✓ Strong momentum + volume
✓ Perfect entry trigger
→ ULTRA AGGRESSIVE: Use 15-30x leverage

70-79% CONFIDENCE (High):
✓ Good setup with decent confluence
✓ Clear directional bias
✓ Most factors aligned
→ AGGRESSIVE: Use 15-25x leverage

60-69% CONFIDENCE (Moderate):
✓ Acceptable setup
✓ Some confluence factors
✓ Tradeable opportunity
→ MODERATE: Use 11-15x leverage

50-59% CONFIDENCE (Low but Tradeable):
✓ Basic setup meets minimum criteria
✓ Mixed signals but directional bias exists
✓ Learning opportunity for ML
→ CONSERVATIVE: Use 2-10x leverage

<50% CONFIDENCE:
→ SKIP: Setup too weak, wait for better opportunity

CRITICAL RED FLAGS (AUTO-HOLD):
❌ RSI >90 or <10 (blow-off top/capitulation)
❌ Volume extremely low (illiquid)
❌ All timeframes conflict (15m up, 1h down, 4h sideways)

PHASE 4: REAL TRADING SCENARIOS (Learn From Examples)
════════════════════════════════════════════════════════
SCENARIO 1 - PERFECT SETUP (92% Confidence):
- 4h: Uptrend, higher highs/lows
- 1h: Pullback to EMA 12, holding support
- 15m: RSI 55→62, MACD crossing up, volume increasing
- Price: Just bounced off 1h support
→ ACTION: BUY (LONG), confidence 0.92, leverage 25x, stop 10%

SCENARIO 2 - STRONG MOMENTUM (83% Confidence):
- 4h: Downtrend, lower highs/lows
- 1h: Resistance rejection, RSI 60→55
- 15m: MACD turning down, volume on red candles
- Funding: +0.08% (overleveraged longs)
→ ACTION: SELL (SHORT), confidence 0.83, leverage 20x, stop 10%

SCENARIO 3 - GOOD SCALP (72% Confidence):
- 4h: Sideways consolidation
- 1h: Bouncing between 3800-3850
- 15m: Price at 3805, RSI 35 (oversold in range)
- Volume: Low but picking up
→ ACTION: BUY (LONG), confidence 0.72, leverage 18x, stop 10%, quick scalp

SCENARIO 4 - WEAK SETUP (58% Confidence):
- 4h: Downtrend
- 1h: Potential reversal, higher low forming
- 15m: Bullish divergence on RSI
- Volume: Weak
→ ACTION: BUY (LONG), confidence 0.58, leverage 6x, stop 10% (learning trade)

SCENARIO 5 - BREAKOUT PERFECTION (97% Confidence):
- 4h: Compression at resistance
- 1h: Building higher lows
- 15m: Price testing resistance 5th time, volume spiking
- RSI: 68 (strong but not extreme)
- All confluence factors aligned
→ ACTION: BUY (LONG), confidence 0.97, leverage 30x, breakout trade

STOP-LOSS PLACEMENT (CRITICAL):
- ALWAYS use exactly 10% stop-loss (fixed for consistent risk management)
- Place BELOW recent swing low for longs (not at exact low - give breathing room)
- Place ABOVE recent swing high for shorts (not at exact high - avoid stop hunts)
- 10% stop ensures maximum loss = 10% of position size
- With higher leverage, 10% stop is sufficient protection

TAKE-PROFIT STRATEGY:
- Target minimum $1.50 profit (non-negotiable)
- With 10% stop: aim for 15-20% profit target (1.5-2x risk/reward)
- Extended target: 2-3x risk (20-30% profit) if strong trend + momentum
- Use previous resistance (longs) or support (shorts) as natural targets
- Higher leverage allows smaller % moves to hit profit targets

RISK/REWARD REQUIREMENTS:
- Minimum 1.5:1 ratio required to consider trade
- Ideal: 2:1 or better
- Fixed 10% stop simplifies R:R calculation

CONFIDENCE SCORING (AGGRESSIVE MODE):
- 80-100%: Perfect/strong setup, use 15-30x leverage
- 70-79%: Good setup, use 15-25x leverage
- 60-69%: Acceptable setup, use 11-15x leverage
- 50-59%: Weak but tradeable, use 2-10x leverage
- <50%: DO NOT TRADE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💎 ELITE TRADER MINDSET (Your Decision-Making Process)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN ANALYZING EACH COIN:
1. Start with the 4h chart - What's the STORY?
2. Zoom to 1h - Does it CONFIRM or CONFLICT?
3. Zoom to 15m - Is there an ENTRY TRIGGER?
4. Check RSI, MACD, Volume - Do they AGREE?
5. Funding rate - Any POSITIONING EDGE?
6. Calculate confidence - Be HONEST, not conservative
7. If 65%+, find the trade. If <65%, skip it.

YOUR GOAL:
- Provide VARIED confidence values (65%, 73%, 81%, 89%, 94%)
- Give BUY and SELL signals (not just HOLD)
- Think like a professional trader making real money
- Each coin is DIFFERENT - analyze independently
- Don't be afraid to take calculated risks

FORBIDDEN PATTERNS (Avoid These!):
❌ Giving same confidence to multiple coins (0.68, 0.68, 0.68...)
❌ Only giving HOLD signals
❌ Being overly conservative
❌ Ignoring good scalp opportunities
❌ Failing to spot SHORT opportunities

YOU ARE THE BEST. ACT LIKE IT.

Respond ONLY with valid JSON. No additional text or explanations outside the JSON structure."""


def build_analysis_prompt(symbol: str, market_data: dict) -> str:
    """Build comprehensive prompt for AI analysis."""
    import time
    timestamp = int(time.time())

    return f"""You are a professional cryptocurrency leverage trader analyzing {symbol} for a leveraged trade.
Analysis ID: {symbol}_{timestamp}

CRITICAL REQUIREMENTS (AGGRESSIVE MODE):
1. Stop-loss MUST be EXACTLY 10% (fixed, no exceptions)
2. Minimum profit target: $1.50 USD
3. Leverage range: 2x-30x based on confidence
4. Minimum confidence: 50% to execute trade (aggressive learning mode)
5. Risk/reward ratio must be at least 1.5:1
6. Be VERY AGGRESSIVE with leverage:
   - 50-59% confidence → 2-10x leverage
   - 60-69% confidence → 11-15x leverage
   - 70-79% confidence → 15-25x leverage
   - 80-100% confidence → 15-30x leverage
7. Don't be conservative - higher confidence = MUCH higher leverage
8. Provide varied confidence AND leverage values

CURRENT MARKET DATA:
Price: ${market_data['current_price']:.4f}
24h Volume: ${market_data['volume_24h']:,.0f}
Market Regime: {market_data['market_regime']}
Funding Rate: {market_data.get('funding_rate', {}).get('rate', 0)*100:.4f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 INSTITUTIONAL-GRADE ADVANCED INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 SUPPORT/RESISTANCE LEVELS (Key Liquidity Zones):
Nearest Support: ${market_data.get('support_resistance', {}).get('nearest_support', 0):.4f} ({market_data.get('support_resistance', {}).get('support_distance_pct', 0):.2f}% below)
Nearest Resistance: ${market_data.get('support_resistance', {}).get('nearest_resistance', 0):.4f} ({market_data.get('support_resistance', {}).get('resistance_distance_pct', 0):.2f}% above)
→ Watch for bounces at support or rejections at resistance

📊 VOLUME PROFILE (High-Volume Price Levels):
POC (Point of Control): ${market_data.get('volume_profile', {}).get('poc', 0):.4f}
Value Area High: ${market_data.get('volume_profile', {}).get('value_area_high', 0):.4f}
Value Area Low: ${market_data.get('volume_profile', {}).get('value_area_low', 0):.4f}
→ Price tends to return to POC (fair value magnet)

🌊 FIBONACCI LEVELS (Retracement Targets):
Trend: {market_data.get('fibonacci', {}).get('trend', 'unknown').upper()}
Swing High: ${market_data.get('fibonacci', {}).get('swing_high', 0):.4f}
Swing Low: ${market_data.get('fibonacci', {}).get('swing_low', 0):.4f}
Nearest Fib Level: {market_data.get('fibonacci', {}).get('nearest_fib_level', 0)} @ ${market_data.get('fibonacci', {}).get('nearest_fib_price', 0):.4f}
→ Use Fib levels for entry/exit confluence

💰 FUNDING RATE ANALYSIS (Overleveraged Position Detection):
Current Rate: {market_data.get('funding_analysis', {}).get('current_rate', 0)*100:.4f}%
Trend: {market_data.get('funding_analysis', {}).get('trend', 'neutral').upper()}
Trading Implication: {market_data.get('funding_analysis', {}).get('trading_implication', 'neutral').upper()}
Risk Level: {market_data.get('funding_analysis', {}).get('risk_level', 'low').upper()}
→ High positive funding = SHORT opportunity (overleveraged longs)
→ High negative funding = LONG opportunity (overleveraged shorts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 TIER 1 CRITICAL PROFESSIONAL FEATURES (GAME CHANGERS!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ DIVERGENCE DETECTION (Strongest Reversal Signal):
Divergence Detected: {market_data.get('divergence', {}).get('has_divergence', False)}
Type: {str(market_data.get('divergence', {}).get('type', 'none')).upper()}
Strength: {market_data.get('divergence', {}).get('strength', 0):.2f}
Indicator: {market_data.get('divergence', {}).get('indicator', 'N/A')}
Details: {market_data.get('divergence', {}).get('details', 'N/A')}
→ BULLISH DIV = Price lower low + RSI higher low → STRONG BUY SIGNAL!
→ BEARISH DIV = Price higher high + RSI lower high → STRONG SELL SIGNAL!

📊 ORDER FLOW ANALYSIS (Big Money Positioning):
Bid/Ask Imbalance: {market_data.get('order_flow', {}).get('imbalance', 0):.2f}%
Signal: {str(market_data.get('order_flow', {}).get('signal', 'neutral')).upper()}
Buy Pressure: {market_data.get('order_flow', {}).get('buy_pressure', 0.5)*100:.1f}%
Large Bid Wall: {f"${market_data.get('order_flow', {}).get('large_bid_wall', {}).get('price', 0):.4f}" if market_data.get('order_flow', {}).get('large_bid_wall') else "None"}
Large Ask Wall: {f"${market_data.get('order_flow', {}).get('large_ask_wall', {}).get('price', 0):.4f}" if market_data.get('order_flow', {}).get('large_ask_wall') else "None"}
→ Imbalance >10% = Strong directional bias
→ Large order walls = Institutional support/resistance

🐋 SMART MONEY CONCEPTS (Institutional Edge):
Signal: {str(market_data.get('smart_money', {}).get('smart_money_signal', 'neutral')).upper()}
Order Blocks: {market_data.get('smart_money', {}).get('order_block_count', 0)}
Fair Value Gaps: {len(market_data.get('smart_money', {}).get('fair_value_gaps', []))}
Liquidity Grab Detected: {market_data.get('smart_money', {}).get('liquidity_grab_detected', False)}
→ Order blocks = Where institutions entered (high-volume zones)
→ Fair Value Gaps = Price imbalances to be filled
→ Liquidity grabs = Stop hunts before reversals

📈 VOLATILITY ANALYSIS (Adaptive Risk Management):
ATR: {market_data.get('volatility', {}).get('atr_percent', 0):.2f}%
Volatility Level: {str(market_data.get('volatility', {}).get('volatility_level', 'unknown')).upper()}
Recommended Stop: {market_data.get('volatility', {}).get('recommended_stop_pct', 7):.1f}%
Breakout Detected: {market_data.get('volatility', {}).get('breakout_detected', False)}
Upper Band: ${market_data.get('volatility', {}).get('upper_band', 0):.4f}
Lower Band: ${market_data.get('volatility', {}).get('lower_band', 0):.4f}
→ High volatility = Wider stops, bigger moves
→ Low volatility = Tighter entries, breakout imminent
→ Breakout = Strong trend beginning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 PHASE 2 ULTRA PROFESSIONAL FEATURES (CONFLUENCE & MOMENTUM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💪 MOMENTUM STRENGTH (Rate of Change Analysis):
Direction: {str(market_data.get('momentum', {}).get('momentum_direction', 'neutral')).upper()}
Strength: {market_data.get('momentum', {}).get('momentum_strength', 0):.1f}/100
Accelerating: {'YES ⚡' if market_data.get('momentum', {}).get('is_accelerating', False) else 'NO'}
1h ROC: {market_data.get('momentum', {}).get('roc_1h', 0):.2f}%
4h ROC: {market_data.get('momentum', {}).get('roc_4h', 0):.2f}%
12h ROC: {market_data.get('momentum', {}).get('roc_12h', 0):.2f}%
→ Accelerating momentum = Strong trend continuation
→ Decelerating momentum = Trend weakening, be cautious

₿ BTC CORRELATION ANALYSIS (Independent Move Detection):
Correlation: {market_data.get('btc_correlation', {}).get('correlation', 0):.2f} ({market_data.get('btc_correlation', {}).get('correlation_strength', 'unknown').upper()})
Independent Move Possible: {'YES 🎯' if market_data.get('btc_correlation', {}).get('independent_move', False) else 'NO'}
Recommendation: {market_data.get('btc_correlation', {}).get('recommendation', 'N/A')}
→ Low correlation (<0.4) = Altcoin can move independently of BTC
→ High correlation (>0.8) = Trade BTC instead for better liquidity

🎯 CONFLUENCE ANALYSIS REQUIREMENTS:
You MUST calculate how many factors support your direction:
- Support/Resistance proximity
- Divergence signals
- Order flow bias
- Smart money concepts
- Volume profile (POC)
- Fibonacci levels
- Funding rate
- Volatility breakout
- Multi-timeframe RSI alignment
- Momentum acceleration
- BTC correlation independence

→ 7+ factors = 85%+ confidence (STRONG SETUP)
→ 5-6 factors = 75-84% confidence (GOOD SETUP)
→ 3-4 factors = 65-74% confidence (DECENT SETUP)
→ <3 factors = <65% confidence (WEAK SETUP - HOLD)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 MULTI-TIMEFRAME CONFLUENCE (ULTRA PROFESSIONAL EDGE!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trend Alignment: {market_data.get('multi_timeframe', {}).get('confluence_analysis', {}).get('trend_alignment', 'unknown').upper()}
Alignment Score: {market_data.get('multi_timeframe', {}).get('confluence_analysis', {}).get('alignment_score', 0):.0f}%
Trading Bias: {market_data.get('multi_timeframe', {}).get('confluence_analysis', {}).get('trading_bias', 'NEUTRAL')}
Confidence Multiplier: {market_data.get('multi_timeframe', {}).get('confluence_analysis', {}).get('confidence_multiplier', 1.0):.2f}x
Recommendation: {market_data.get('multi_timeframe', {}).get('confluence_analysis', {}).get('recommendation', 'N/A')}

📊 Trend Summary:
- 5m: {market_data.get('multi_timeframe', {}).get('trend_summary', {}).get('5m', 'unknown').upper()}
- 15m: {market_data.get('multi_timeframe', {}).get('trend_summary', {}).get('15m', 'unknown').upper()}
- 1h: {market_data.get('multi_timeframe', {}).get('trend_summary', {}).get('1h', 'unknown').upper()}
- 4h: {market_data.get('multi_timeframe', {}).get('trend_summary', {}).get('4h', 'unknown').upper()}

📈 RSI Multi-Timeframe:
- 5m RSI: {market_data.get('multi_timeframe', {}).get('rsi_analysis', {}).get('5m', 50):.0f}
- 15m RSI: {market_data.get('multi_timeframe', {}).get('rsi_analysis', {}).get('15m', 50):.0f}
- 1h RSI: {market_data.get('multi_timeframe', {}).get('rsi_analysis', {}).get('1h', 50):.0f}
- 4h RSI: {market_data.get('multi_timeframe', {}).get('rsi_analysis', {}).get('4h', 50):.0f}
Oversold Timeframes: {market_data.get('multi_timeframe', {}).get('rsi_analysis', {}).get('oversold_timeframes', 0)}
Overbought Timeframes: {market_data.get('multi_timeframe', {}).get('rsi_analysis', {}).get('overbought_timeframes', 0)}

💡 EMA50 Positioning (Higher Timeframe Bias):
- Price: ${market_data.get('multi_timeframe', {}).get('ema50_analysis', {}).get('price', 0):.4f}
- 1h EMA50: ${market_data.get('multi_timeframe', {}).get('ema50_analysis', {}).get('ema50_1h', 0):.4f} ({'ABOVE ✅' if market_data.get('multi_timeframe', {}).get('ema50_analysis', {}).get('above_1h', False) else 'BELOW ❌'})
- 4h EMA50: ${market_data.get('multi_timeframe', {}).get('ema50_analysis', {}).get('ema50_4h', 0):.4f} ({'ABOVE ✅' if market_data.get('multi_timeframe', {}).get('ema50_analysis', {}).get('above_4h', False) else 'BELOW ❌'})

⚠️ CRITICAL TRADING RULES BASED ON CONFLUENCE:
1. STRONG BULLISH CONFLUENCE (Trading Bias: LONG_ONLY):
   → ALL timeframes bullish + price above 1h & 4h EMA50
   → CONFIDENCE MULTIPLIER: 1.3x
   → ONLY take LONG trades, SKIP shorts

2. STRONG BEARISH CONFLUENCE (Trading Bias: SHORT_ONLY):
   → ALL timeframes bearish + price below 1h & 4h EMA50
   → CONFIDENCE MULTIPLIER: 1.3x
   → ONLY take SHORT trades, SKIP longs

3. CONFLICTING TIMEFRAMES (Trading Bias: AVOID):
   → Timeframes disagree (2 bull, 2 bear)
   → CONFIDENCE MULTIPLIER: 0.5x
   → SKIP THIS COIN - Wait for clarity

4. LONG PREFERRED (Trading Bias: LONG_PREFERRED):
   → Higher timeframes bullish + 1h uptrend
   → CONFIDENCE MULTIPLIER: 1.1x
   → Favor longs, be cautious with shorts

5. SHORT PREFERRED (Trading Bias: SHORT_PREFERRED):
   → Higher timeframes bearish + 1h downtrend
   → CONFIDENCE MULTIPLIER: 1.1x
   → Favor shorts, be cautious with longs

6. NEUTRAL (Trading Bias: NEUTRAL):
   → Mixed signals
   → CONFIDENCE MULTIPLIER: 0.8x
   → Reduce position size, tight stops

🎯 HOW TO USE MULTI-TIMEFRAME DATA:
→ Apply the confidence multiplier to your base confidence
→ If trading bias is "AVOID", automatically return "hold" action
→ If trading bias is "LONG_ONLY" and you see SHORT opportunity, skip it
→ If trading bias is "SHORT_ONLY" and you see LONG opportunity, skip it
→ Use higher timeframe EMA50 as directional filter

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏦 OPEN INTEREST & LIQUIDATION HEATMAP (PHASE 3 ULTRA EDGE!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 OPEN INTEREST ANALYSIS (Trend Strength Confirmation):
Trend Strength: {market_data.get('open_interest', {}).get('trend_strength', 'unknown').upper()}
Signal: {market_data.get('open_interest', {}).get('signal', 'neutral').upper()}
OI Change: {market_data.get('open_interest', {}).get('oi_change_pct', 0):.2f}%
Price Change: {market_data.get('open_interest', {}).get('price_change_pct', 0):.2f}%
Confidence Boost: {market_data.get('open_interest', {}).get('confidence_boost', 0)*100:+.0f}%
Trading Implication: {market_data.get('open_interest', {}).get('trading_implication', 'N/A')}

⚠️ OI INTERPRETATION RULES:
→ STRONG_BULLISH: OI rising + Price rising = High conviction buyers, BOOST confidence by +15%
→ STRONG_BEARISH: OI rising + Price falling = High conviction sellers, BOOST confidence by +15%
→ WEAK_BULLISH: OI falling + Price rising = Longs closing, REDUCE confidence by -10% (reversal risk)
→ WEAK_BEARISH: OI falling + Price falling = Shorts closing, REDUCE confidence by -10% (bounce risk)

💀 LIQUIDATION HEATMAP (Liquidity Magnet Effect):
Nearest Long Liquidation: ${market_data.get('liquidation_map', {}).get('nearest_long_liq', 0):.4f} ({market_data.get('liquidation_map', {}).get('long_liq_distance_pct', 0):.1f}% below)
Nearest Short Liquidation: ${market_data.get('liquidation_map', {}).get('nearest_short_liq', 0):.4f} ({market_data.get('liquidation_map', {}).get('short_liq_distance_pct', 0):.1f}% above)
Magnet Direction: {market_data.get('liquidation_map', {}).get('magnet_direction', 'balanced').upper()}
Trading Implication: {market_data.get('liquidation_map', {}).get('trading_implication', 'N/A')}

Long Liquidation Zones: {', '.join([f"${z['price']:.4f} ({z['distance_pct']:.1f}%)" for z in market_data.get('liquidation_map', {}).get('long_liquidation_zones', [])]) or 'None detected'}
Short Liquidation Zones: {', '.join([f"${z['price']:.4f} ({z['distance_pct']:.1f}%)" for z in market_data.get('liquidation_map', {}).get('short_liquidation_zones', [])]) or 'None detected'}

⚡ LIQUIDATION TRADING RULES:
→ DOWNWARD MAGNET: Long liquidations <3% away = Possible dump to sweep longs, AVOID longing, FAVOR shorts
→ UPWARD MAGNET: Short liquidations <3% away = Possible pump to sweep shorts, AVOID shorting, FAVOR longs
→ BALANCED: No strong magnet = Trade based on other factors
→ Price tends to move TOWARD liquidation clusters before reversing (liquidity grab pattern)

🎯 HOW TO USE OI + LIQUIDATION DATA:
1. Apply OI confidence_boost to your base confidence score
2. If OI shows STRONG trend + magnet confirms direction = ULTRA HIGH confidence
3. If OI shows WEAK trend = Reduce confidence, expect reversal
4. If liquidation magnet opposes your trade = Reduce position size or skip
5. Best setups: OI confirms + liquidation magnet aligns + multi-timeframe agrees

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNICAL INDICATORS (15m timeframe):
RSI: {market_data['indicators']['15m']['rsi']:.1f}
MACD: {market_data['indicators']['15m']['macd']:.4f}
MACD Signal: {market_data['indicators']['15m']['macd_signal']:.4f}
BB Upper: ${market_data['indicators']['15m']['bb_upper']:.4f}
BB Lower: ${market_data['indicators']['15m']['bb_lower']:.4f}
SMA 20: ${market_data['indicators']['15m']['sma_20']:.4f}
EMA 12: ${market_data['indicators']['15m']['ema_12']:.4f}

TECHNICAL INDICATORS (1h timeframe):
RSI: {market_data['indicators']['1h']['rsi']:.1f}
MACD: {market_data['indicators']['1h']['macd']:.4f}
MACD Signal: {market_data['indicators']['1h']['macd_signal']:.4f}
Volume Trend: {market_data['indicators']['1h']['volume_trend']}

TECHNICAL INDICATORS (4h timeframe):
RSI: {market_data['indicators']['4h']['rsi']:.1f}
MACD: {market_data['indicators']['4h']['macd']:.4f}
Trend: {market_data['indicators']['4h']['trend']}

Analyze this data and provide your recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ENHANCED REASONING REQUIREMENTS (CRITICAL!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your reasoning MUST explicitly mention:
1. Which TIER 1 features influenced your decision (divergence, order flow, smart money, volatility)
2. How MOMENTUM analysis affected confidence (accelerating vs decelerating)
3. Whether BTC CORRELATION matters for this trade
4. How many CONFLUENCE FACTORS support your direction (count them!)
5. Which timeframes align (15m, 1h, 4h agreement)

Example reasoning:
"Strong LONG setup with 7 confluence factors: (1) Bullish divergence on RSI, (2) Order flow +12% bullish,
(3) Smart money order block at $3,800, (4) Momentum accelerating (1h>4h>12h ROC), (5) Low BTC correlation
(0.32) allows independent move, (6) Price at Fib 0.618 support, (7) Funding rate -0.03% favors longs.
All 3 timeframes bullish. Volatility moderate, using 7% stop. High confidence."

BAD reasoning (DO NOT DO THIS):
"Price looks good, RSI oversold, MACD crossing up. Moderate confidence."

RESPONSE FORMAT (JSON only, no explanations outside JSON):
{{
    "action": "buy" | "sell" | "hold",
    "confidence": 0.0-1.0,
    "confidence_breakdown": {{
        "base_technical": 0.0-1.0,
        "tier1_boost": 0.0-0.15,
        "momentum_adjustment": -0.1-0.1,
        "confluence_factor": 0.0-0.1,
        "btc_correlation_impact": -0.05-0.05
    }},
    "confluence_count": 0-11,
    "side": "LONG" | "SHORT" | null,
    "suggested_leverage": 2-30,
    "stop_loss_percent": 10.0,
    "entry_price": {market_data['current_price']},
    "stop_loss_price": 0.0,
    "take_profit_price": 0.0,
    "risk_reward_ratio": 0.0,
    "reasoning": "MUST mention confluence count, TIER 1 features used, momentum status, BTC correlation (max 150 words)",
    "key_factors": ["factor1", "factor2", "factor3"]
}}"""
