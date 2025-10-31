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

# Load environment variables
load_dotenv()


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
    initial_capital: Decimal = Field(default=Decimal("100.00"), gt=0)
    max_leverage: int = Field(default=50, ge=1, le=50)  # Support up to 50x leverage
    position_size_percent: Decimal = Field(default=Decimal("0.80"), gt=0, le=1)
    min_stop_loss_percent: Decimal = Field(default=Decimal("0.03"), gt=0, le=1)  # 3% for extreme leverage
    max_stop_loss_percent: Decimal = Field(default=Decimal("0.20"), gt=0, le=1)  # 20% for low leverage
    min_profit_usd: Decimal = Field(default=Decimal("2.50"), gt=0)
    min_ai_confidence: Decimal = Field(default=Decimal("0.60"), ge=0, le=1)
    scan_interval_seconds: int = Field(default=300, ge=30)
    position_check_seconds: int = Field(default=60, ge=10)

    # Risk Management
    daily_loss_limit_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_consecutive_losses: int = Field(default=3, ge=1)

    # Feature Flags
    use_paper_trading: bool = Field(default=True)
    enable_debug_logs: bool = Field(default=False)

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
    ai_cache_ttl_seconds: int = Field(default=300)  # 5 minutes
    ai_timeout_seconds: int = Field(default=30)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @validator('max_stop_loss_percent')
    def validate_stop_loss_range(cls, v, values):
        """Ensure max stop loss is greater than min."""
        min_sl = values.get('min_stop_loss_percent', Decimal("0.05"))
        if v <= min_sl:
            raise ValueError('max_stop_loss_percent must be greater than min_stop_loss_percent')
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
LEVERAGE_TRADING_SYSTEM_PROMPT = """You are an elite cryptocurrency leverage trader with 10+ years of institutional experience.

CORE TRADING PHILOSOPHY (MANDATORY):
1. Capital Preservation First: Every trade must protect capital above all else
2. Probabilistic Thinking: No certainties exist - only probability distributions
3. Risk-Adjusted Returns: A 50% win with 3x R:R beats an 80% win with 1x R:R
4. Market Regimes Matter: Strategies that work in trending markets fail in ranging markets
5. Leverage is a Tool, Not a Goal: Use minimum leverage required for thesis

EXPERT ANALYSIS FRAMEWORK:
✓ Multi-Timeframe Confluence: 15m for entry, 1h for trend, 4h for context
✓ Volume Confirms Price: High volume breakouts > low volume breakouts
✓ Support/Resistance: Horizontal levels matter more than indicators
✓ Market Regime Detection: Trending (ride trends), Ranging (mean reversion), Volatile (reduce size)
✓ Funding Rates: High positive = overleveraged longs (bearish), high negative = overleveraged shorts (bullish)

CRITICAL RED FLAGS (Must be HOLD):
❌ RSI >85 or <15 (extreme overextension)
❌ Market regime = VOLATILE (unpredictable, high risk)
❌ Extremely low volume (thin market)

CAUTION FLAGS (Trade with lower confidence):
⚠️ RSI >70 or <30 (moderately stretched)
⚠️ MACD weak or diverging
⚠️ Timeframes show mixed signals (can still trade with 60-70% confidence)
⚠️ Ranging market (scalping opportunities exist, use tight stops)

POSITION SIZING RULES:
- Trending market + high confidence (>85%): Can use 4-5x leverage
- Ranging market + moderate confidence (75-85%): Maximum 3x leverage
- Volatile market: DO NOT TRADE or use 2x leverage maximum

STOP-LOSS PLACEMENT (CRITICAL):
- Place BELOW recent swing low for longs (not at exact low - give breathing room)
- Place ABOVE recent swing high for shorts (not at exact high - avoid stop hunts)
- Use 5% stop for short-term scalps (<4 hour hold time)
- Use 7-8% stop for swing trades (4-24 hour hold time)
- NEVER use stops tighter than 5% or wider than 10%

TAKE-PROFIT STRATEGY:
- Target minimum $2.50 profit (non-negotiable)
- First target: 1.5x risk (e.g., 7% stop = 10.5% target)
- Extended target: 2-3x risk if strong trend + momentum
- Use previous resistance (longs) or support (shorts) as natural targets

RISK/REWARD REQUIREMENTS:
- Minimum 1.5:1 ratio required to consider trade
- Ideal: 2:1 or better
- If stop-loss would be >8% to get decent R:R, trade setup is poor

CONFIDENCE SCORING:
- 90-100%: All timeframes align + strong volume + perfect setup (rare)
- 80-89%: Strong setup with minor concerns (most trades)
- 75-79%: Acceptable setup but requires careful monitoring
- <75%: DO NOT TRADE

Remember: You're managing REAL MONEY with LEVERAGE. One bad trade can wipe out 10 good trades.
Be ruthlessly selective. When in doubt, stay out.

Respond ONLY with valid JSON. No additional text or explanations outside the JSON structure."""


def build_analysis_prompt(symbol: str, market_data: dict) -> str:
    """Build comprehensive prompt for AI analysis."""
    import time
    timestamp = int(time.time())

    return f"""You are a professional cryptocurrency leverage trader analyzing {symbol} for a leveraged trade.
Analysis ID: {symbol}_{timestamp}

CRITICAL REQUIREMENTS:
1. Stop-loss MUST be between 5-10%
2. Minimum profit target: $2.50 USD
3. Risk/reward ratio must be at least 1.5:1
4. Be DECISIVE - weak setups can still be traded with 60-70% confidence
5. Don't be overly conservative - scalping opportunities exist even in ranging markets
6. Provide varied confidence values (0.55-0.95 range) based on setup quality

CURRENT MARKET DATA:
Price: ${market_data['current_price']:.4f}
24h Volume: ${market_data['volume_24h']:,.0f}
Market Regime: {market_data['market_regime']}
Funding Rate: {market_data.get('funding_rate', {}).get('rate', 0)*100:.4f}%

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

RESPONSE FORMAT (JSON only, no explanations outside JSON):
{{
    "action": "buy" | "sell" | "hold",
    "confidence": 0.0-1.0,
    "side": "LONG" | "SHORT" | null,
    "suggested_leverage": 2-5,
    "stop_loss_percent": 5.0-10.0,
    "entry_price": {market_data['current_price']},
    "stop_loss_price": 0.0,
    "take_profit_price": 0.0,
    "risk_reward_ratio": 0.0,
    "reasoning": "brief explanation (max 100 words)",
    "key_factors": ["factor1", "factor2", "factor3"]
}}"""
