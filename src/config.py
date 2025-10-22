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

    # AI API Keys
    claude_api_key: str = Field(..., min_length=1)
    deepseek_api_key: str = Field(..., min_length=1)
    grok_api_key: Optional[str] = None

    # Telegram Configuration
    telegram_bot_token: str = Field(..., min_length=1)
    telegram_chat_id: str = Field(..., min_length=1)

    # Database Configuration
    database_url: str = Field(..., min_length=1)
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Trading Configuration
    initial_capital: Decimal = Field(default=Decimal("100.00"), gt=0)
    max_leverage: int = Field(default=5, ge=1, le=10)
    position_size_percent: Decimal = Field(default=Decimal("0.80"), gt=0, le=1)
    min_stop_loss_percent: Decimal = Field(default=Decimal("0.05"), gt=0, le=1)
    max_stop_loss_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    min_profit_usd: Decimal = Field(default=Decimal("2.50"), gt=0)
    min_ai_confidence: Decimal = Field(default=Decimal("0.75"), ge=0, le=1)
    scan_interval_seconds: int = Field(default=300, ge=30)
    position_check_seconds: int = Field(default=60, ge=10)

    # Risk Management
    daily_loss_limit_percent: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_consecutive_losses: int = Field(default=3, ge=1)

    # Feature Flags
    use_paper_trading: bool = Field(default=True)
    enable_grok: bool = Field(default=False)
    enable_debug_logs: bool = Field(default=False)

    # Trading Symbols (high liquidity perpetual futures)
    trading_symbols: list[str] = Field(default=[
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'SOL/USDT:USDT',
        'BNB/USDT:USDT',
        'XRP/USDT:USDT',
        'DOGE/USDT:USDT',
        'ADA/USDT:USDT',
        'AVAX/USDT:USDT',
        'MATIC/USDT:USDT',
        'DOT/USDT:USDT'
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
LEVERAGE_TRADING_SYSTEM_PROMPT = """You are an expert cryptocurrency leverage trader with years of experience.

Your trading philosophy:
- Capital preservation is priority #1
- Only take high-probability trades
- Always use stop-losses
- Risk/reward must favor reward
- Be skeptical and cautious with leverage
- Avoid overtrading

You specialize in:
- Multi-timeframe technical analysis
- Risk management for leveraged positions
- Identifying high-probability setups
- Detecting market regime changes

You NEVER:
- Recommend trades without clear edge
- Ignore stop-losses
- Take excessive risk
- Trade in unclear market conditions

Respond ONLY with valid JSON. No additional text or explanations outside the JSON structure."""


def build_analysis_prompt(symbol: str, market_data: dict) -> str:
    """Build comprehensive prompt for AI analysis."""
    return f"""You are a professional cryptocurrency leverage trader analyzing {symbol} for a leveraged trade.

CRITICAL REQUIREMENTS:
1. Stop-loss MUST be between 5-10%
2. Minimum profit target: $2.50 USD
3. Risk/reward ratio must be at least 1.5:1
4. Only recommend trades with 75%+ confidence
5. Consider this is LEVERAGE trading - be conservative

CURRENT MARKET DATA:
Price: ${market_data['current_price']:.4f}
24h Volume: ${market_data['volume_24h']:,.0f}
Market Regime: {market_data['market_regime']}
Funding Rate: {market_data['funding_rate']['rate']*100:.4f}%

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
