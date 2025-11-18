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
    initial_capital: Decimal = Field(default=Decimal("2.29"), gt=0)  # 🔥 SYNCED: Current actual Binance balance
    min_leverage: int = Field(default=10, ge=1, le=50)  # 🎯 USER: 10x leverage (reduced risk)
    max_leverage: int = Field(default=10, ge=1, le=50)  # 🎯 USER: 10x leverage (reduced risk)
    max_concurrent_positions: int = Field(default=2, ge=1, le=30)  # 🔧 USER: 2 positions max (bakiye/2 = her pozisyon için yarısı)
    position_size_percent: Decimal = Field(default=Decimal("0.85"), gt=0, le=1)  # 85% per position
    min_stop_loss_percent: Decimal = Field(default=Decimal("17.0"), gt=0, le=100)  # 🎯 USER: $5 loss at 10x leverage (~17% of margin)
    max_stop_loss_percent: Decimal = Field(default=Decimal("34.0"), gt=0, le=100)  # 🎯 USER: $10 loss at 10x leverage (~34% of margin)
    min_profit_usd: Decimal = Field(default=Decimal("2.50"), gt=0)  # 🎯 USER: $2.50 profit target
    max_position_hours: int = Field(default=8, ge=1, le=48)  # Auto-close after 8h
    min_ai_confidence: Decimal = Field(default=Decimal("0.80"), ge=0, le=1)  # 🔥 STRICT: 80% min confidence (prevents low-quality entries and immediate SL hits)
    scan_interval_seconds: int = Field(default=20, ge=10)  # 🔥 ULTRA AGGRESSIVE: 20 seconds for rapid AI+ML learning
    position_check_seconds: int = Field(default=3, ge=1)  # 🔥 INSTANT PROFIT CAPTURE: 3 seconds for real-time monitoring (prevent profit erosion!)

    # Risk Management
    daily_loss_limit_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_consecutive_losses: int = Field(default=5, ge=1)  # Log warning after 5 losses (but continue trading for ML learning)

    # Feature Flags
    use_paper_trading: bool = Field(default=True)
    enable_debug_logs: bool = Field(default=False)
    enable_short_trades: bool = Field(default=True)  # Enable SHORT trades for complete ML learning
    enable_ml_exit: bool = Field(default=False)  # ML exit DISABLED - rely only on stop-loss/take-profit/trailing (USER REQUEST: positions closing too fast)

    # 🚀 PROFESSIONAL TRADING FEATURES (12 Improvements for 90-95% Win Rate!)
    enable_time_filter: bool = Field(default=True)  # ✅ ENABLED: Time-based trading filter (avoid toxic hours)
    enable_trailing_stop: bool = Field(default=True)  # ✅ ENABLED: Trailing stop-loss system
    enable_partial_exits: bool = Field(default=False)  # 3-tier partial exit system (DISABLED: creates orders below Binance $20 minimum)
    enable_market_regime: bool = Field(default=False)  # Market regime detection
    enable_multi_timeframe: bool = Field(default=False)  # Multi-timeframe confluence analysis
    enable_dynamic_position_sizing: bool = Field(default=False)  # Kelly Criterion + quality-based sizing
    enable_news_filter: bool = Field(default=False)  # News/event filter (avoid high-impact news)
    enable_ml_ensemble: bool = Field(default=False)  # ML ensemble (multiple models voting)
    enable_smc_patterns: bool = Field(default=False)  # Smart Money Concepts (order blocks, FVG)
    enable_order_flow: bool = Field(default=False)  # Order flow analysis (bid/ask imbalance)
    enable_whale_tracking: bool = Field(default=False)  # Whale activity tracking (PLACEHOLDER)
    enable_online_learning: bool = Field(default=False)  # Online learning (adaptive ML updates)

    # Trading Symbols (high liquidity perpetual futures) - 120 coins (USER REQUEST: 100-150)
    trading_symbols: list[str] = Field(default=[
        # Top 10 - Highest Market Cap (Ultra Liquid)
        'BTC/USDT:USDT',    # Bitcoin
        'ETH/USDT:USDT',    # Ethereum
        'SOL/USDT:USDT',    # Solana
        'BNB/USDT:USDT',    # Binance Coin
        'XRP/USDT:USDT',    # Ripple
        'DOGE/USDT:USDT',   # Dogecoin
        'ADA/USDT:USDT',    # Cardano
        'AVAX/USDT:USDT',   # Avalanche
        'TON/USDT:USDT',    # Toncoin
        'TRX/USDT:USDT',    # Tron

        # 11-30: Major Altcoins
        'LINK/USDT:USDT',   # Chainlink
        'UNI/USDT:USDT',    # Uniswap
        'ATOM/USDT:USDT',   # Cosmos
        'DOT/USDT:USDT',    # Polkadot
        'LTC/USDT:USDT',    # Litecoin
        'BCH/USDT:USDT',    # Bitcoin Cash
        'NEAR/USDT:USDT',   # Near Protocol
        'ICP/USDT:USDT',    # Internet Computer
        'FIL/USDT:USDT',    # Filecoin
        'APT/USDT:USDT',    # Aptos
        'SUI/USDT:USDT',    # Sui
        'ARB/USDT:USDT',    # Arbitrum
        'OP/USDT:USDT',     # Optimism
        'POL/USDT:USDT',    # Polygon
        'INJ/USDT:USDT',    # Injective
        'FTM/USDT:USDT',    # Fantom
        'AAVE/USDT:USDT',   # Aave
        'MKR/USDT:USDT',    # Maker
        'GRT/USDT:USDT',    # The Graph
        'RUNE/USDT:USDT',   # THORChain

        # 31-50: DeFi & Layer 2
        'STX/USDT:USDT',    # Stacks
        'IMX/USDT:USDT',    # Immutable X
        'FET/USDT:USDT',    # Fetch.ai
        'ALGO/USDT:USDT',   # Algorand
        'VET/USDT:USDT',    # VeChain
        'HBAR/USDT:USDT',   # Hedera
        'ETC/USDT:USDT',    # Ethereum Classic
        'XLM/USDT:USDT',    # Stellar
        'SAND/USDT:USDT',   # Sandbox
        'MANA/USDT:USDT',   # Decentraland
        'AXS/USDT:USDT',    # Axie Infinity
        'GALA/USDT:USDT',   # Gala
        'ENJ/USDT:USDT',    # Enjin
        'CHZ/USDT:USDT',    # Chiliz
        'FLOW/USDT:USDT',   # Flow
        'THETA/USDT:USDT',  # Theta
        # 'EOS/USDT:USDT',    # EOS - DELISTED from Binance
        'KLAY/USDT:USDT',   # Klaytn
        'XTZ/USDT:USDT',    # Tezos
        'ZEC/USDT:USDT',    # Zcash

        # 51-70: Mid-Cap Gems
        'DASH/USDT:USDT',   # Dash
        'COMP/USDT:USDT',   # Compound
        'SNX/USDT:USDT',    # Synthetix
        'CRV/USDT:USDT',    # Curve
        'SUSHI/USDT:USDT',  # SushiSwap
        '1INCH/USDT:USDT',  # 1inch
        'BAL/USDT:USDT',    # Balancer
        'YFI/USDT:USDT',    # Yearn Finance
        'ZRX/USDT:USDT',    # 0x
        'LRC/USDT:USDT',    # Loopring
        # 'RNDR/USDT:USDT',   # Render - Symbol changed to RENDER/USDT
        'AR/USDT:USDT',     # Arweave
        'ROSE/USDT:USDT',   # Oasis Network
        'ONE/USDT:USDT',    # Harmony
        'ZIL/USDT:USDT',    # Zilliqa
        'CELO/USDT:USDT',   # Celo
        'QTUM/USDT:USDT',   # Qtum
        'WAVES/USDT:USDT',  # Waves
        'ICX/USDT:USDT',    # ICON
        'ONT/USDT:USDT',    # Ontology

        # 71-90: High Volume Trading Pairs
        'PEPE/USDT:USDT',   # Pepe (Meme)
        'SHIB/USDT:USDT',   # Shiba Inu
        'WIF/USDT:USDT',    # Dogwifhat
        'FLOKI/USDT:USDT',  # Floki
        'BONK/USDT:USDT',   # Bonk
        'PEOPLE/USDT:USDT', # ConstitutionDAO
        # 'LUNC/USDT:USDT',   # Terra Classic - DELISTED from Binance
        'LUNA/USDT:USDT',   # Terra
        'SEI/USDT:USDT',    # Sei
        'TIA/USDT:USDT',    # Celestia
        'ORDI/USDT:USDT',   # Ordinals
        'BLUR/USDT:USDT',   # Blur
        'WLD/USDT:USDT',    # Worldcoin
        # 'MATIC/USDT:USDT',  # Matic - Symbol changed to POL/USDT
        # 'CRO/USDT:USDT',    # Cronos - DELISTED from Binance
        'FTT/USDT:USDT',    # FTX Token
        'GMT/USDT:USDT',    # STEPN
        'APE/USDT:USDT',    # ApeCoin
        'LDO/USDT:USDT',    # Lido DAO
        'CFX/USDT:USDT',    # Conflux

        # 91-110: Emerging & Volatile
        'MASK/USDT:USDT',   # Mask Network
        'MINA/USDT:USDT',   # Mina Protocol
        'SKL/USDT:USDT',    # SKALE
        'DYDX/USDT:USDT',   # dYdX
        'GMX/USDT:USDT',    # GMX
        'PERP/USDT:USDT',   # Perpetual Protocol
        'STORJ/USDT:USDT',  # Storj
        # 'AUDIO/USDT:USDT',  # Audius - DELISTED from Binance
        'C98/USDT:USDT',    # Coin98
        'ALICE/USDT:USDT',  # My Neighbor Alice
        'TLM/USDT:USDT',    # Alien Worlds
        'SLP/USDT:USDT',    # Smooth Love Potion
        'DENT/USDT:USDT',   # Dent
        'HOT/USDT:USDT',    # Holo
        'COTI/USDT:USDT',   # COTI
        'OCEAN/USDT:USDT',  # Ocean Protocol
        'ANKR/USDT:USDT',   # Ankr
        'BNT/USDT:USDT',    # Bancor
        'REN/USDT:USDT',    # Ren
        'BAND/USDT:USDT',   # Band Protocol

        # 111-120: Additional High Liquidity
        'XMR/USDT:USDT',    # Monero
        'IOTA/USDT:USDT',   # IOTA
        'NEO/USDT:USDT',    # Neo
        'EGLD/USDT:USDT',   # MultiversX
        'KSM/USDT:USDT',    # Kusama
        # 'RETH/USDT:USDT',   # Rocket Pool ETH - DELISTED from Binance
        'ETHW/USDT:USDT',   # EthereumPoW
        'USTC/USDT:USDT',   # TerraClassicUSD
        'JASMY/USDT:USDT',  # JasmyCoin
        'UNFI/USDT:USDT'    # Unifi Protocol
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
CONFIDENCE SCORING SYSTEM (20x FIXED LEVERAGE - ALL TRADES):

85-100% CONFIDENCE (Ultra High):
✓ All timeframes perfectly aligned
✓ 8+ confluence factors
✓ Strong momentum + volume + no major risks
✓ Perfect entry trigger at key level
→ FIXED LEVERAGE: Use 20x leverage

75-84% CONFIDENCE (High):
✓ Strong setup with 6-7 confluence factors
✓ Clear directional bias
✓ Most factors aligned
→ FIXED LEVERAGE: Use 20x leverage

65-74% CONFIDENCE (Good):
✓ Good setup with 4-5 confluence factors
✓ Clear direction
✓ Solid opportunity
→ FIXED LEVERAGE: Use 20x leverage

55-64% CONFIDENCE (Acceptable):
✓ Decent setup with 3-4 confluence factors
✓ Some directional bias
✓ ML learning opportunity
→ FIXED LEVERAGE: Use 20x leverage

50-54% CONFIDENCE (Speculative):
✓ Weak setup with 2-3 confluence factors
✓ Slight edge detected by AI
✓ Pure ML learning trade
→ FIXED LEVERAGE: Use 20x leverage

<50% CONFIDENCE:
→ SKIP: No edge detected
→ CRITICAL: Minimum 50% confidence required (MAXIMUM AGGRESSIVE MODE)

⚠️ CRITICAL RULE - ACTION REQUIREMENTS:
✅ If confidence ≥ 50% → MUST return action="buy" or "sell" with side="LONG" or "SHORT"
❌ If confidence ≥ 50% → NEVER return action="hold" or side=null
❌ Only return "hold" if confidence <50% (no edge detected)
→ Even weak setups (50-54%) MUST have direction (slight bullish=buy, slight bearish=sell)

CRITICAL RED FLAGS (AUTO-HOLD with confidence <50%):
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
→ ACTION: BUY (LONG), confidence 0.92, leverage 9x, stop 16%

SCENARIO 2 - STRONG MOMENTUM (83% Confidence):
- 4h: Downtrend, lower highs/lows
- 1h: Resistance rejection, RSI 60→55
- 15m: MACD turning down, volume on red candles
- Funding: +0.08% (overleveraged longs)
→ ACTION: SELL (SHORT), confidence 0.83, leverage 7x, stop 15%

SCENARIO 3 - GOOD SCALP (72% Confidence):
- 4h: Sideways consolidation
- 1h: Bouncing between 3800-3850
- 15m: Price at 3805, RSI 35 (oversold in range)
- Volume: Low but picking up
→ ACTION: BUY (LONG), confidence 0.72, leverage 4x, stop 14%, quick scalp

SCENARIO 4 - ACCEPTABLE SETUP (62% Confidence):
- 4h: Downtrend
- 1h: Potential reversal, higher low forming
- 15m: Bullish divergence on RSI
- Volume: Weak but picking up
→ ACTION: BUY (LONG), confidence 0.62, leverage 4x, tight stop 14% (ML learning trade)

SCENARIO 4B - SPECULATIVE SETUP (52% Confidence - Maximum Learning):
- 4h: Sideways
- 1h: Mixed signals
- 15m: Slight bullish bias, RSI 52
- Volume: Average
→ ACTION: BUY (LONG), confidence 0.52, leverage 3x, very tight stop 12% (pure ML test)

SCENARIO 5 - BREAKOUT PERFECTION (92% Confidence):
- 4h: Compression at resistance
- 1h: Building higher lows
- 15m: Price testing resistance 5th time, volume spiking
- RSI: 68 (strong but not extreme)
- All confluence factors aligned
→ ACTION: BUY (LONG), confidence 0.92, leverage 10x (max), breakout trade

STOP-LOSS PLACEMENT (20x FIXED LEVERAGE - BALANCED APPROACH):
- Stop-loss range: 4-5% of position value (BALANCED for 20x leverage)
- With 20x leverage, this translates to 0.20-0.25% price movement
- Maximum risk per trade: ~$1 loss on $20 position (4-5% of position)
- Place BELOW recent swing low for longs (minimal breathing room)
- Place ABOVE recent swing high for shorts (accept some stop-hunt risk)
- Tight stops required for 20x leverage to keep liquidation far enough
- 20x leverage = balanced risk/reward with good profit potential

TAKE-PROFIT STRATEGY:
- Target minimum $1.20-$1.50 GROSS profit (covers commission+slippage for $1.00 NET profit)
- Binance commission: ~0.04-0.10% total (open+close)
- Market order slippage: ~0.10-0.50%
- Total overhead: ~$0.15-$0.50 per trade
- With 4-5% stop: aim for 5-7% profit target (1.2:1 risk/reward minimum)
- Extended target: 1.5-2x risk (8-12% profit) if strong trend + momentum
- Use previous resistance (longs) or support (shorts) as natural targets
- 20x leverage = small % moves generate good profit

RISK/REWARD REQUIREMENTS:
- Minimum 1.5:1 ratio required to consider trade
- Ideal: 2:1 or better
- Adaptive stop-loss based on win rate and volatility

CONFIDENCE SCORING (20x FIXED LEVERAGE - ALL TRADES):
- 85-100%: Perfect setup, use 20x leverage (max risk $1 per trade)
- 75-84%: Strong setup, use 20x leverage (max risk $1 per trade)
- 65-74%: Good setup, use 20x leverage (max risk $1 per trade)
- 55-64%: Acceptable setup, use 20x leverage (max risk $1 per trade)
- 50-54%: Speculative setup, use 20x leverage (max risk $1 per trade)
- <50%: DO NOT TRADE (no edge detected)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💎 ELITE TRADER MINDSET (Your Decision-Making Process)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN ANALYZING EACH COIN:
1. Start with the 4h chart - What's the STORY?
2. Zoom to 1h - Does it CONFIRM or CONFLICT?
3. Zoom to 15m - Is there an ENTRY TRIGGER?
4. Check RSI, MACD, Volume - Do they AGREE?
5. Funding rate - Any POSITIONING EDGE?
6. Calculate confidence - Be HONEST and THOROUGH
7. If 50%+, find the trade. If <50%, skip it (MAXIMUM AGGRESSIVE MODE - AI+ML FULL TEST).

YOUR GOAL:
- Provide VARIED confidence values (50%, 58%, 66%, 74%, 82%, 90%, 95%)
- Give BUY and SELL signals when ANY edge detected (50%+)
- Maximum ML learning from diverse setups
- Each coin is DIFFERENT - analyze independently
- MAXIMUM LEARNING - take calculated risks even on weak setups (50-54%)

FORBIDDEN PATTERNS (Avoid These!):
❌ Giving same confidence to multiple coins (0.52, 0.52, 0.52...)
❌ Only giving HOLD signals (be VERY aggressive in finding setups!)
❌ Trading with confidence below 50% (no edge = no trade)
❌ Using ANY leverage other than 20x (FIXED at 20x always!)
❌ Ignoring confluence factors
❌ **CRITICAL:** Returning action="hold" when confidence ≥ 50%
❌ **CRITICAL:** Returning side=null when confidence ≥ 50%

⚠️ MANDATORY RULE - READ CAREFULLY:
If your calculated confidence is ≥ 50%, you MUST:
1. Set action = "buy" OR "sell" (never "hold")
2. Set side = "LONG" OR "SHORT" (never null)
3. Even if setup is weak (50-54%), pick a direction based on:
   - Slight RSI bias? (>50 = bullish/LONG, <50 = bearish/SHORT)
   - Order flow? (positive = LONG, negative = SHORT)
   - MACD trend? (positive = LONG, negative = SHORT)
   - Strategy recommendation? (buy @ X% = LONG, sell @ Y% = SHORT)

Example:
- If confidence=0.52 and RSI=52 → action="buy", side="LONG"
- If confidence=0.52 and RSI=48 → action="sell", side="SHORT"
- If confidence=0.49 → action="hold", side=null (OK, below threshold)

YOU ARE THE BEST. ACT LIKE IT.

Respond ONLY with valid JSON. No additional text or explanations outside the JSON structure."""


def build_analysis_prompt(symbol: str, market_data: dict) -> str:
    """Build comprehensive prompt for AI analysis."""
    import time
    timestamp = int(time.time())

    return f"""You are a professional cryptocurrency leverage trader analyzing {symbol} for a leveraged trade.
Analysis ID: {symbol}_{timestamp}

CRITICAL REQUIREMENTS (20x FIXED LEVERAGE - ALL TRADES):
1. Stop-loss range: 4-5% of position value (balanced for 20x leverage)
2. Minimum profit target: $1.20-$1.50 USD GROSS (covers commission+slippage for $1.00 NET profit)
3. Leverage: ALWAYS 20x (FIXED - USER REQUEST)
4. Minimum confidence: 50% to execute trade
5. Risk/reward ratio must be at least 1.2:1 (accounting for commission+slippage)
6. LEVERAGE RULES:
   - ALL trades use 20x leverage (no variation)
   - NEVER exceed 20x leverage (hard limit)
   - NEVER go below 20x leverage (hard limit)
7. Stop-loss must ensure max $1 loss per trade (4-5% of position)
8. Commission awareness: Add ~$0.20-$0.50 to profit target to cover fees
9. Provide varied confidence (50-95%) but ALWAYS 20x leverage

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

→ 8+ factors = 85%+ confidence (ULTRA STRONG SETUP - TRADE)
→ 6-7 factors = 75-84% confidence (STRONG SETUP - TRADE)
→ 4-5 factors = 65-74% confidence (GOOD SETUP - TRADE)
→ 3 factors = 55-64% confidence (ACCEPTABLE SETUP - TRADE)
→ 2 factors = 50-54% confidence (SPECULATIVE SETUP - TRADE with 3x leverage only)
→ <2 factors = <50% confidence (NO EDGE - HOLD)

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

⚠️ CRITICAL VALIDATION RULES BEFORE RESPONDING:
1. If confidence ≥ 0.50 → action MUST be "buy" or "sell" (NEVER "hold")
2. If confidence ≥ 0.50 → side MUST be "LONG" or "SHORT" (NEVER null)
3. If confidence < 0.50 → action MUST be "hold" and side MUST be null
4. Use RSI/MACD/Order Flow as tiebreaker for weak setups (50-54%)

{{
    "action": "buy" | "sell" | "hold",  // ⚠️ "hold" ONLY if confidence < 0.50
    "confidence": 0.0-1.0,
    "confidence_breakdown": {{
        "base_technical": 0.0-1.0,
        "tier1_boost": 0.0-0.15,
        "momentum_adjustment": -0.1-0.1,
        "confluence_factor": 0.0-0.1,
        "btc_correlation_impact": -0.05-0.05
    }},
    "confluence_count": 0-11,
    "side": "LONG" | "SHORT" | null,  // ⚠️ null ONLY if confidence < 0.50
    "suggested_leverage": 20,  // ⚠️ ALWAYS 20 (FIXED - USER REQUEST)
    "stop_loss_percent": 4.0-5.0,  // ⚠️ 4-5% for 20x leverage
    "entry_price": {market_data['current_price']},
    "stop_loss_price": 0.0,
    "take_profit_price": 0.0,
    "risk_reward_ratio": 0.0,
    "reasoning": "MUST mention confluence count, TIER 1 features used, momentum status, BTC correlation (max 150 words)",
    "key_factors": ["factor1", "factor2", "factor3"]
}}"""
