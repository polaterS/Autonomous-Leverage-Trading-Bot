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
    # 🔥 LIVE TRADING CONFIG: $100 balance, conservative sizing
    initial_capital: Decimal = Field(default=Decimal("100.00"), gt=0)  # 🔥 ACTUAL: $100 Binance balance
    min_leverage: int = Field(default=20, ge=1, le=50)  # 🚀 AGGRESSIVE: 20x min leverage
    max_leverage: int = Field(default=20, ge=1, le=50)  # 🔥 FIXED: 20x max (no dynamic)
    max_concurrent_positions: int = Field(default=2, ge=1, le=30)  # 🎯 USER REQUEST: Max 2 positions at a time
    position_size_percent: Decimal = Field(default=Decimal("0.40"), gt=0, le=1)  # 🔥 SAFE: 40% = $40 margin per position
    min_stop_loss_percent: Decimal = Field(default=Decimal("2.0"), gt=0, le=100)  # 🚀 AGGRESSIVE: 2% min for 25x leverage (tight stop = $37.5 loss on $1,875 position)
    max_stop_loss_percent: Decimal = Field(default=Decimal("3.0"), gt=0, le=100)  # 🚀 AGGRESSIVE: 3% max for 25x leverage ($56.25 max loss = 37.5% of capital)
    min_profit_usd: Decimal = Field(default=Decimal("5.0"), gt=0)  # 🎯 USER REQUEST: $5 min profit (realistic for $50 margin positions)
    max_position_hours: int = Field(default=8, ge=1, le=48)  # Auto-close after 8h
    min_ai_confidence: Decimal = Field(default=Decimal("0.60"), ge=0, le=1)  # 🧪 TEST MODE: 60% min confidence (loosened from 70% for more opportunities)
    scan_interval_seconds: int = Field(default=60, ge=10)  # 🚀 FAST ENTRY: 1 minute scan (catch trends early!)
    position_check_seconds: int = Field(default=15, ge=1)  # 🔥 REAL-TIME: 15 seconds for profit/loss monitoring

    # Risk Management
    daily_loss_limit_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_consecutive_losses: int = Field(default=5, ge=1)  # Log warning after 5 losses (but continue trading for ML learning)
    position_cooldown_minutes: int = Field(default=30, ge=0)  # 🚫 COOLDOWN: Wait X minutes before re-trading same symbol (prevents doubling down, 0=disabled)

    # Feature Flags
    use_paper_trading: bool = Field(default=True)
    enable_debug_logs: bool = Field(default=False)
    enable_short_trades: bool = Field(default=True)  # Enable SHORT trades for complete ML learning
    enable_ml_exit: bool = Field(default=False)  # ML exit DISABLED - rely only on stop-loss/take-profit/trailing (USER REQUEST: positions closing too fast)

    # 🚀 PROFESSIONAL TRADING FEATURES - 75% ACCURACY TARGET
    enable_time_filter: bool = Field(default=False)  # 🔥 DISABLED: 24/7 trading enabled (USER REQUEST: Trust AI analysis at all hours)
    enable_trailing_stop: bool = Field(default=True)  # ✅ ENABLED: Trailing stop-loss system
    enable_partial_exits: bool = Field(default=False)  # 3-tier partial exit system (DISABLED: creates orders below Binance $20 minimum)
    enable_market_regime: bool = Field(default=False)  # Market regime detection
    enable_multi_timeframe: bool = Field(default=True)  # 🎯 CRITICAL: Multi-timeframe confluence (MANDATORY for 75% accuracy)
    enable_dynamic_position_sizing: bool = Field(default=False)  # Kelly Criterion + quality-based sizing
    enable_news_filter: bool = Field(default=False)  # News/event filter (avoid high-impact news)
    enable_ml_ensemble: bool = Field(default=False)  # ML ensemble (multiple models voting)
    enable_smc_patterns: bool = Field(default=True)  # 🎯 CRITICAL: Smart Money Concepts - market structure confirmation
    enable_order_flow: bool = Field(default=True)  # 🎯 CRITICAL: Order flow analysis - buyer/seller pressure confirmation
    enable_whale_tracking: bool = Field(default=False)  # Whale activity tracking (PLACEHOLDER)
    enable_online_learning: bool = Field(default=False)  # Online learning (adaptive ML updates)
    enable_realtime_detection: bool = Field(default=True)  # 🚀 INSTANT ENTRY: WebSocket-based real-time trend detection (< 1 sec entry!)

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
        # 'PEPE/USDT:USDT',   # Pepe (Meme) - NO FUTURES on Binance
        # 'SHIB/USDT:USDT',   # Shiba Inu - NO FUTURES on Binance
        'WIF/USDT:USDT',    # Dogwifhat
        # 'FLOKI/USDT:USDT',  # Floki - NO FUTURES on Binance
        # 'BONK/USDT:USDT',   # Bonk - NO FUTURES on Binance
        'PEOPLE/USDT:USDT', # ConstitutionDAO
        # 'LUNC/USDT:USDT',   # Terra Classic - DELISTED from Binance
        # 'LUNA/USDT:USDT',   # Terra - NO FUTURES on Binance (LUNC available but delisted)
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🚀 PROFESSIONAL TRADING SYSTEM - 75-80% WIN RATE TARGET
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Enhanced Trading System Configuration
    enable_enhanced_system: bool = Field(default=True)  # ✅ Use professional enhanced trading system

    # Confluence Scoring (Quality Filter)
    enable_confluence_filtering: bool = Field(default=True)  # ✅ Filter trades by quality score
    min_confluence_score: int = Field(default=60, ge=0, le=100)  # 🎯 60+ confluence with MTF+momentum+volume checks
    confluence_weights: dict = Field(default={
        'multi_timeframe': 25,
        'volume_profile': 20,
        'indicators': 20,
        'market_regime': 15,
        'support_resistance': 15,
        'risk_reward': 5
    })

    # Volume Profile Analysis
    enable_volume_profile: bool = Field(default=True)  # ✅ Use volume-based S/R levels
    volume_profile_bins: int = Field(default=50, ge=20, le=100)  # Price bins for volume distribution
    volume_hvn_percentile: int = Field(default=75, ge=50, le=95)  # High Volume Node threshold
    volume_lvn_percentile: int = Field(default=25, ge=5, le=50)  # Low Volume Node threshold

    # Dynamic Position Sizing
    enable_dynamic_sizing: bool = Field(default=True)  # ✅ Adjust size based on trade quality
    base_risk_percent: Decimal = Field(default=Decimal("0.02"), gt=0, le=0.10)  # 2% base risk
    max_risk_percent: Decimal = Field(default=Decimal("0.05"), gt=0, le=0.15)  # 5% maximum risk per trade
    min_risk_percent: Decimal = Field(default=Decimal("0.005"), gt=0, le=0.05)  # 0.5% minimum risk

    # Kelly Criterion (Optional)
    use_kelly_criterion: bool = Field(default=True)  # ✅ Use Kelly for optimal sizing
    kelly_fraction: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)  # 1/4 Kelly (conservative)

    # Quality-Based Size Multipliers
    size_multiplier_excellent: Decimal = Field(default=Decimal("1.8"), ge=1, le=3)  # 90-100 score
    size_multiplier_strong: Decimal = Field(default=Decimal("1.4"), ge=1, le=2)  # 80-89 score
    size_multiplier_good: Decimal = Field(default=Decimal("1.0"), ge=1, le=1.5)  # 75-79 score

    # Drawdown Protection
    reduce_size_on_drawdown: bool = Field(default=True)  # ✅ Reduce sizes during drawdown
    drawdown_reduce_threshold: Decimal = Field(default=Decimal("0.10"), gt=0, le=0.50)  # -10% threshold
    drawdown_reduce_severe: Decimal = Field(default=Decimal("0.20"), gt=0, le=0.50)  # -20% severe
    profit_scale_threshold: Decimal = Field(default=Decimal("0.30"), gt=0, le=1.0)  # +30% scale up

    # Dynamic Profit Targets
    enable_dynamic_targets: bool = Field(default=True)  # ✅ ATR-based dynamic targets
    min_rr_ratio: Decimal = Field(default=Decimal("3.0"), ge=1, le=10)  # Minimum 3:1 R/R
    tp1_r_multiple: Decimal = Field(default=Decimal("1.5"), gt=0, le=5)  # TP1 at +1.5R
    tp2_r_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=10)  # TP2 at +3R
    tp3_r_multiple: Decimal = Field(default=Decimal("5.0"), gt=0, le=15)  # TP3 at +5R
    tp1_allocation: Decimal = Field(default=Decimal("0.40"), gt=0, le=1)  # 40% at TP1
    tp2_allocation: Decimal = Field(default=Decimal("0.40"), gt=0, le=1)  # 40% at TP2
    tp3_allocation: Decimal = Field(default=Decimal("0.20"), gt=0, le=1)  # 20% runner
    atr_multiplier_base: Decimal = Field(default=Decimal("2.0"), gt=0, le=5)  # 2x ATR for targets

    # Advanced Trailing Stop
    enable_advanced_trailing: bool = Field(default=True)  # ✅ 4-stage progressive trailing
    trailing_break_even_r: Decimal = Field(default=Decimal("1.5"), gt=0, le=5)  # Move to BE at +1.5R
    trailing_50pct_r: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)  # 50% trail at +2.5R
    trailing_25pct_r: Decimal = Field(default=Decimal("4.0"), gt=0, le=15)  # 25% trail at +4R
    trailing_atr_r: Decimal = Field(default=Decimal("6.0"), gt=0, le=20)  # ATR trail at +6R
    trailing_atr_multiplier: Decimal = Field(default=Decimal("1.5"), gt=0, le=5)  # 1.5x ATR distance
    trailing_momentum_aware: bool = Field(default=True)  # ✅ Tighten on weak momentum

    # Market Regime Filters
    enable_regime_filter: bool = Field(default=True)  # ✅ Don't trade in choppy markets
    skip_choppy_markets: bool = Field(default=True)  # ✅ Skip CHOPPY regime entirely
    min_regime_confidence: Decimal = Field(default=Decimal("0.55"), ge=0, le=1)  # 55% min confidence

    # Performance Tracking
    track_confluence_scores: bool = Field(default=True)  # ✅ Log all scores for analysis
    track_quality_distribution: bool = Field(default=True)  # ✅ Track EXCELLENT/STRONG/GOOD counts
    calculate_rolling_metrics: bool = Field(default=True)  # ✅ Rolling win rate, avg R/R
    rolling_window_trades: int = Field(default=20, ge=10, le=100)  # Last 20 trades for rolling metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
CONFIDENCE SCORING SYSTEM (3x-5x DYNAMIC LEVERAGE - ULTRA CONSERVATIVE):

85-100% CONFIDENCE (Ultra High):
✓ All timeframes perfectly aligned
✓ 8+ confluence factors
✓ Strong momentum + volume + no major risks
✓ Perfect entry trigger at key level
→ DYNAMIC LEVERAGE: Use 5x leverage (maximum)

75-84% CONFIDENCE (High):
✓ Strong setup with 6-7 confluence factors
✓ Clear directional bias
✓ Most factors aligned
→ DYNAMIC LEVERAGE: Use 4x leverage

65-74% CONFIDENCE (Good):
✓ Good setup with 4-5 confluence factors
✓ Clear direction
✓ Solid opportunity
→ DYNAMIC LEVERAGE: Use 4x leverage

55-64% CONFIDENCE (Acceptable):
✓ Decent setup with 3-4 confluence factors
✓ Some directional bias
✓ ML learning opportunity
→ DYNAMIC LEVERAGE: Use 3x leverage (minimum)

50-54% CONFIDENCE (Speculative):
✓ Weak setup with 2-3 confluence factors
✓ Slight edge detected by AI
✓ Pure ML learning trade
→ DYNAMIC LEVERAGE: Use 3x leverage (minimum)

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
→ ACTION: BUY (LONG), confidence 0.92, leverage 5x, stop 16%

SCENARIO 2 - STRONG MOMENTUM (83% Confidence):
- 4h: Downtrend, lower highs/lows
- 1h: Resistance rejection, RSI 60→55
- 15m: MACD turning down, volume on red candles
- Funding: +0.08% (overleveraged longs)
→ ACTION: SELL (SHORT), confidence 0.83, leverage 5x, stop 15%

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
→ ACTION: BUY (LONG), confidence 0.92, leverage 5x (max), breakout trade

STOP-LOSS PLACEMENT (3x-5x DYNAMIC LEVERAGE - ULTRA CONSERVATIVE APPROACH):
- Stop-loss range: 12-18% of position value (ATR-based for 3x-5x leverage)
- With 3x-5x leverage, this translates to 2.4-6% price movement
- Maximum risk per trade: ~$0.70-1.47 loss on smaller positions (12-18% of position)
- Place BELOW recent swing low for longs (ATR-based buffer)
- Place ABOVE recent swing high for shorts (ATR-based buffer)
- Dynamic stops adapt to volatility (tight in low vol, wider in high vol)
- 3x-5x leverage = ultra conservative risk/reward with sustainable growth

TAKE-PROFIT STRATEGY:
- Target minimum $1.20-$1.50 GROSS profit (covers commission+slippage for $1.00 NET profit)
- Binance commission: ~0.04-0.10% total (open+close)
- Market order slippage: ~0.10-0.50%
- Total overhead: ~$0.15-$0.50 per trade
- With 4-5% stop: aim for 5-7% profit target (1.2:1 risk/reward minimum)
- Extended target: 1.5-2x risk (8-12% profit) if strong trend + momentum
- Use previous resistance (longs) or support (shorts) as natural targets
- 3x-5x leverage = conservative approach with steady profits

RISK/REWARD REQUIREMENTS:
- Minimum 1.5:1 ratio required to consider trade
- Ideal: 2:1 or better
- Adaptive stop-loss based on win rate and volatility

CONFIDENCE SCORING (3x-5x DYNAMIC LEVERAGE - ULTRA CONSERVATIVE):
- 85-100%: Perfect setup, use 5x leverage (max risk ~$1.47 per trade)
- 75-84%: Strong setup, use 4x leverage (max risk ~$1.26 per trade)
- 65-74%: Good setup, use 4x leverage (max risk ~$1.05 per trade)
- 55-64%: Acceptable setup, use 3x leverage (max risk ~$1.05 per trade)
- 50-54%: Speculative setup, use 3x leverage (max risk ~$1.05 per trade)
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

CRITICAL REQUIREMENTS (3x-5x DYNAMIC LEVERAGE - ULTRA CONSERVATIVE):
1. Stop-loss range: 12-18% of position value (ATR-based for 3x-5x leverage)
2. Minimum profit target: $1.20-$1.50 USD GROSS (covers commission+slippage for $1.00 NET profit)
3. Leverage: DYNAMIC 3x-5x based on confidence (ULTRA CONSERVATIVE APPROACH)
4. Minimum confidence: 50% to execute trade
5. Risk/reward ratio must be at least 1.2:1 (accounting for commission+slippage)
6. LEVERAGE RULES:
   - 85-100% confidence: 5x leverage (maximum)
   - 75-84% confidence: 4x leverage
   - 50-74% confidence: 3x leverage (minimum)
   - NEVER exceed 5x leverage (hard limit)
   - NEVER go below 3x leverage (hard limit)
7. Stop-loss must ensure max ~$0.70-1.47 loss per trade (12-18% of position, ATR-based)
8. Commission awareness: Add ~$0.10-$0.30 to profit target to cover fees
9. Provide varied confidence (50-95%) with dynamic leverage (3x-5x)

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
