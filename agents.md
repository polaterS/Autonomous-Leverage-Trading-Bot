# agents.md - Autonomous Leverage Trading Bot with Telegram Notifications

## Project Overview

Build a **fully autonomous cryptocurrency leverage trading bot** that continuously analyzes markets using multiple AI models (DeepSeek, Claude, Grok), automatically executes trades with strict risk management rules, and sends real-time notifications via Telegram. The user simply monitors their portfolio through Telegram notifications without any manual intervention.

---

## Core Requirements

### Functional Requirements

**Autonomous Trading Engine:**
- Continuously scan 10-15 high-liquidity cryptocurrency pairs every 5-15 minutes
- Use AI consensus (minimum 2 out of 3 AI models must agree) to identify trading opportunities
- Automatically open leveraged positions (2x-5x) when high-confidence signals are detected
- Execute trades without human approval or intervention
- Immediately scan for new opportunities after closing a position

**Mandatory Risk Management Rules:**
- **CRITICAL:** Every single trade MUST have a stop-loss between 5-10%
- **CRITICAL:** Never close a profitable position until profit exceeds $2.50 USD minimum
- **CRITICAL:** Auto-close if stop-loss is hit (5-10% loss)
- **CRITICAL:** Auto-close if liquidation distance falls below 5%
- Position size: Maximum 80% of available capital per trade
- Maximum leverage: 5x (configurable, but recommended 2x-3x for beginners)
- Daily loss circuit breaker: Stop trading if daily loss exceeds 10% of capital
- Consecutive loss limit: Pause trading after 3 consecutive losing trades

**AI Analysis Requirements:**
- Integrate three AI models: Claude 4.5 Sonnet, DeepSeek V3, and Grok (optional)
- Multi-timeframe analysis: 15m, 1h, 4h charts
- Technical indicators: RSI, MACD, Bollinger Bands, Volume, Moving Averages
- Market regime detection: Trending, Ranging, or Volatile conditions
- Require AI consensus: At least 2 out of 3 models must agree on direction
- Minimum confidence threshold: 75% for any trade execution
- Analyze funding rates for perpetual futures

**Telegram Notification System:**
- Real-time notifications for every trade event
- Position opened: Symbol, side (LONG/SHORT), leverage, entry price, stop-loss, liquidation price
- Position monitoring: Unrealized P&L updates every 5 minutes while position is open
- Position closed: Exit price, realized P&L (in USD and %), reason for closing, trade duration
- Portfolio updates: Current capital, daily P&L, win rate
- Market scans: Best opportunity found, AI confidence score
- Alerts: Circuit breaker activations, errors, critical warnings
- Daily summary: Total trades, wins/losses, P&L, capital change
- Weekly summary: Performance metrics, best/worst trades

### Non-Functional Requirements

**Performance:**
- Market data latency: < 2 seconds
- AI analysis completion: < 10 seconds per symbol
- Trade execution speed: < 1 second
- Position monitoring frequency: Every 60 seconds when position is open
- Market scanning frequency: Every 300 seconds (5 minutes) when no position

**Reliability:**
- 99.5% uptime target
- Automatic error recovery and retry mechanisms
- Emergency position closure on critical failures
- Graceful degradation if one AI model fails (use remaining models)
- Persistent state storage to survive restarts

**Security:**
- Encrypted storage of exchange API keys and secrets
- API keys with withdrawal restrictions (trading only)
- Secure Telegram bot token management
- Input validation and sanitization
- Rate limiting for API calls

**Scalability:**
- Support single user initially (can be extended to multi-user)
- Efficient AI API usage through intelligent caching
- Optimized exchange API calls to minimize rate limit issues

**Cost Optimization:**
- Intelligent AI caching: Cache analysis results for 3-5 minutes during position monitoring
- Selective AI usage: Use cheaper models (DeepSeek) for routine checks, premium models (Claude) for critical decisions
- Reduce analysis frequency when no position is open
- Target monthly AI cost: $30-60 for moderate trading activity

---

## System Architecture

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                     TELEGRAM INTERFACE                          │
│  User receives notifications & monitors portfolio               │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────┐
│                  AUTONOMOUS TRADING ENGINE                      │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐   │
│  │   Market     │  │   Position    │  │   Notification   │   │
│  │   Scanner    │  │   Monitor     │  │   Manager        │   │
│  │  (5 min)     │  │  (1 min)      │  │   (Real-time)    │   │
│  └──────┬───────┘  └───────┬───────┘  └──────────────────┘   │
│         │                   │                                   │
└─────────┼───────────────────┼───────────────────────────────────┘
          │                   │
┌─────────▼───────────────────▼───────────────────────────────────┐
│                    AI CONSENSUS ENGINE                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │  DeepSeek   │  │    Claude    │  │      Grok        │      │
│  │   Client    │  │    Client    │  │     Client       │      │
│  └─────────────┘  └──────────────┘  └──────────────────┘      │
│                    (Require 2/3 Agreement)                       │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                   RISK MANAGEMENT LAYER                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • 5-10% Stop-Loss Enforcement                           │    │
│  │ • $2.50 Minimum Profit Requirement                      │    │
│  │ • Liquidation Distance Monitoring (5% buffer)           │    │
│  │ • Daily Loss Circuit Breaker (10% max)                  │    │
│  │ • Consecutive Loss Limiter (3 losses → pause)           │    │
│  │ • Position Size Controller (80% max capital)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                    EXCHANGE INTEGRATION                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Binance Futures API (via CCXT)                          │   │
│  │  • Market Data WebSocket                                 │   │
│  │  • Order Execution                                       │   │
│  │  • Position Management                                   │   │
│  │  • Leverage Configuration                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐     │
│  │  PostgreSQL  │  │     Redis      │  │   File Storage  │     │
│  │  (Trades,    │  │  (Cache, AI    │  │  (Logs, State)  │     │
│  │   Positions) │  │   Responses)   │  │                 │     │
│  └──────────────┘  └────────────────┘  └─────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend Runtime
```yaml
Language: Python 3.11+
Async Framework: asyncio (native async/await)
Task Scheduling: APScheduler or Celery Beat
Process Management: systemd or Docker container with restart policy
```

### Core Libraries
```yaml
Exchange Integration: ccxt==4.2.0 (async_support)
AI APIs:
  - anthropic==0.18.1 (Claude)
  - openai==1.12.0 (for DeepSeek/Grok via OpenAI-compatible endpoints)
Telegram: python-telegram-bot==20.7
Data Analysis: pandas==2.2.0, numpy==1.26.3
Technical Indicators: ta==0.11.0 or pandas-ta
HTTP Client: aiohttp==3.9.3
Environment: python-dotenv==1.0.0
```

### Database & Caching
```yaml
Primary Database: PostgreSQL 15+ (trades, positions, logs)
Cache Layer: Redis 7+ (AI response caching, rate limiting)
Alternative (Lightweight): SQLite + in-memory cache (for single-user deployment)
```

### Infrastructure
```yaml
Deployment Options:
  - Option 1: DigitalOcean Droplet ($6-12/month) - VPS with Docker
  - Option 2: Raspberry Pi 4 (8GB) at home - one-time $60-80
  - Option 3: AWS EC2 t3.micro (with free tier or $10/month)

Monitoring:
  - Logging: Python logging module → file rotation
  - Error Tracking: Sentry (free tier) optional
  - Uptime: UptimeRobot (free) monitoring the bot's health endpoint

Containerization:
  - Docker + Docker Compose for easy deployment
  - Auto-restart on failure
  - Volume mounts for persistent data
```

---

## Database Schema

### Core Tables

```sql
-- Trading Configuration
CREATE TABLE trading_config (
    id SERIAL PRIMARY KEY,
    initial_capital DECIMAL(20, 8) NOT NULL,
    current_capital DECIMAL(20, 8) NOT NULL,
    position_size_percent DECIMAL(5, 2) DEFAULT 0.80,
    min_stop_loss_percent DECIMAL(5, 2) DEFAULT 0.05,
    max_stop_loss_percent DECIMAL(5, 2) DEFAULT 0.10,
    min_profit_usd DECIMAL(10, 2) DEFAULT 2.50,
    max_leverage INTEGER DEFAULT 5,
    min_ai_confidence DECIMAL(3, 2) DEFAULT 0.75,
    daily_loss_limit_percent DECIMAL(5, 2) DEFAULT 0.10,
    max_consecutive_losses INTEGER DEFAULT 3,
    is_trading_enabled BOOLEAN DEFAULT true,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Active Positions (should only ever have 0 or 1 row)
CREATE TABLE active_position (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'LONG' or 'SHORT'
    leverage INTEGER NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    position_value_usd DECIMAL(20, 8) NOT NULL,
    stop_loss_price DECIMAL(20, 8) NOT NULL,
    stop_loss_percent DECIMAL(5, 2) NOT NULL,
    min_profit_target_usd DECIMAL(10, 2) NOT NULL,
    min_profit_price DECIMAL(20, 8) NOT NULL,
    liquidation_price DECIMAL(20, 8) NOT NULL,
    unrealized_pnl_usd DECIMAL(20, 8) DEFAULT 0,
    exchange_order_id VARCHAR(100),
    stop_loss_order_id VARCHAR(100),
    ai_model_consensus VARCHAR(100), -- 'claude+deepseek', 'all_three', etc.
    ai_confidence DECIMAL(3, 2),
    entry_time TIMESTAMP DEFAULT NOW(),
    last_check_time TIMESTAMP DEFAULT NOW(),
    CONSTRAINT only_one_position UNIQUE (id)
);

-- Trade History
CREATE TABLE trade_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL,
    side VARCHAR(10) NOT NULL,
    leverage INTEGER NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    position_value_usd DECIMAL(20, 8) NOT NULL,
    realized_pnl_usd DECIMAL(20, 8) NOT NULL,
    pnl_percent DECIMAL(10, 4) NOT NULL,
    stop_loss_percent DECIMAL(5, 2),
    close_reason VARCHAR(100), -- 'stop_loss', 'take_profit', 'ai_signal', 'emergency'
    trade_duration_seconds INTEGER,
    ai_model_consensus VARCHAR(100),
    ai_confidence DECIMAL(3, 2),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP DEFAULT NOW(),
    is_winner BOOLEAN
);

CREATE INDEX idx_trade_history_time ON trade_history(exit_time DESC);
CREATE INDEX idx_trade_history_symbol ON trade_history(symbol);

-- AI Analysis Cache
CREATE TABLE ai_analysis_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL,
    ai_model VARCHAR(20) NOT NULL, -- 'claude', 'deepseek', 'grok'
    timeframe VARCHAR(10) NOT NULL, -- '15m', '1h', '4h'
    analysis_json JSONB NOT NULL,
    confidence DECIMAL(3, 2),
    action VARCHAR(10), -- 'buy', 'sell', 'hold'
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_ai_cache_lookup ON ai_analysis_cache(symbol, ai_model, timeframe, created_at DESC);
CREATE INDEX idx_ai_cache_expiry ON ai_analysis_cache(expires_at);

-- Daily Performance Tracking
CREATE TABLE daily_performance (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    starting_capital DECIMAL(20, 8),
    ending_capital DECIMAL(20, 8),
    daily_pnl DECIMAL(20, 8),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    largest_win DECIMAL(20, 8),
    largest_loss DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT NOW()
);

-- System Logs & Alerts
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20), -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component VARCHAR(50), -- 'scanner', 'position_monitor', 'ai_engine', etc.
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_logs_time ON system_logs(created_at DESC);
CREATE INDEX idx_logs_level ON system_logs(log_level);

-- Circuit Breaker Events
CREATE TABLE circuit_breaker_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50), -- 'daily_loss_limit', 'consecutive_losses', 'liquidation_risk'
    trigger_value DECIMAL(20, 8),
    threshold_value DECIMAL(20, 8),
    action_taken VARCHAR(100),
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Detailed Component Design

### 1. Autonomous Trading Engine

**Main Event Loop:**

```python
"""
Autonomous Trading Engine - Main Loop
Runs continuously, manages market scanning and position monitoring
"""

class AutonomousTradingEngine:
    def __init__(self):
        self.config = TradingConfig()
        self.market_scanner = MarketScanner()
        self.position_monitor = PositionMonitor()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        self.is_running = True
        
    async def run_forever(self):
        """Main infinite loop"""
        await self.notifier.send_startup_message()
        
        while self.is_running:
            try:
                # Check if trading is enabled
                if not await self.check_trading_enabled():
                    await asyncio.sleep(300)
                    continue
                
                # Get current position status
                active_position = await self.get_active_position()
                
                if active_position:
                    # We have an open position - monitor it
                    await self.position_monitor.check_position(active_position)
                    await asyncio.sleep(60)  # Check every 1 minute
                else:
                    # No position - scan for opportunities
                    await self.market_scanner.scan_and_execute()
                    await asyncio.sleep(300)  # Scan every 5 minutes
                    
            except Exception as e:
                await self.handle_critical_error(e)
                await asyncio.sleep(60)
        
    async def check_trading_enabled(self) -> bool:
        """Check circuit breakers and trading status"""
        # Daily loss limit
        daily_pnl = await self.get_daily_pnl()
        if daily_pnl < -(self.config.initial_capital * self.config.daily_loss_limit_percent):
            await self.notifier.send_alert(
                'critical',
                f'Daily loss limit reached: ${daily_pnl:.2f}\nTrading suspended until tomorrow.'
            )
            return False
        
        # Consecutive losses
        consecutive_losses = await self.get_consecutive_losses()
        if consecutive_losses >= self.config.max_consecutive_losses:
            await self.notifier.send_alert(
                'warning',
                f'{consecutive_losses} consecutive losses. Trading paused for review.'
            )
            return False
        
        return True
```

### 2. Market Scanner

**Intelligent Symbol Selection & Analysis:**

```python
"""
Market Scanner - Finds best trading opportunities
Scans multiple symbols, gets AI consensus, ranks opportunities
"""

class MarketScanner:
    def __init__(self):
        self.exchange = ExchangeClient()
        self.ai_engine = AIConsensusEngine()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        
        # Symbols to scan (high liquidity perpetual futures)
        self.symbols = [
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
        ]
    
    async def scan_and_execute(self):
        """
        Main scanning function:
        1. Scan all symbols
        2. Get AI analysis for each
        3. Rank opportunities
        4. Execute best trade if meets criteria
        """
        
        logger.info("Starting market scan...")
        await self.notifier.send_alert('info', '🔍 Market scan started...')
        
        opportunities = []
        
        for symbol in self.symbols:
            try:
                # Get market data
                market_data = await self.gather_market_data(symbol)
                
                # Get AI consensus
                ai_analysis = await self.ai_engine.get_consensus(symbol, market_data)
                
                # Calculate opportunity score
                score = self.calculate_opportunity_score(ai_analysis, market_data)
                
                if score >= 75:  # Minimum 75/100 score
                    opportunities.append({
                        'symbol': symbol,
                        'analysis': ai_analysis,
                        'score': score,
                        'market_data': market_data
                    })
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        if opportunities:
            best = opportunities[0]
            await self.notifier.send_scan_result(
                best['symbol'],
                best['analysis']['confidence'],
                best['analysis']['action']
            )
            
            # Execute trade if score is high enough
            if best['score'] >= 80:
                await self.execute_trade(best)
            else:
                await self.notifier.send_alert(
                    'info',
                    f"Best opportunity: {best['symbol']} (score: {best['score']})\n"
                    f"Not strong enough to trade yet. Waiting for better setup..."
                )
        else:
            await self.notifier.send_alert(
                'info',
                '😐 No good opportunities found. Will scan again in 5 minutes.'
            )
    
    async def gather_market_data(self, symbol: str) -> dict:
        """
        Gather comprehensive market data for AI analysis
        """
        # OHLCV data (multiple timeframes)
        ohlcv_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
        ohlcv_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
        ohlcv_4h = await self.exchange.fetch_ohlcv(symbol, '4h', limit=50)
        
        # Current ticker
        ticker = await self.exchange.fetch_ticker(symbol)
        
        # Funding rate
        funding_rate = await self.exchange.fetch_funding_rate(symbol)
        
        # Technical indicators
        indicators_15m = calculate_indicators(ohlcv_15m)
        indicators_1h = calculate_indicators(ohlcv_1h)
        indicators_4h = calculate_indicators(ohlcv_4h)
        
        # Market regime detection
        regime = detect_market_regime(ohlcv_1h)
        
        return {
            'symbol': symbol,
            'current_price': ticker['last'],
            'volume_24h': ticker['quoteVolume'],
            'ohlcv': {
                '15m': ohlcv_15m[-20:],  # Last 20 candles
                '1h': ohlcv_1h[-20:],
                '4h': ohlcv_4h[-20:]
            },
            'indicators': {
                '15m': indicators_15m,
                '1h': indicators_1h,
                '4h': indicators_4h
            },
            'funding_rate': funding_rate,
            'market_regime': regime  # 'TRENDING', 'RANGING', 'VOLATILE'
        }
    
    def calculate_opportunity_score(self, ai_analysis: dict, market_data: dict) -> float:
        """
        Score opportunity from 0-100 based on multiple factors
        """
        score = 0
        
        # AI Confidence (40 points)
        score += ai_analysis['confidence'] * 40
        
        # AI Consensus strength (20 points)
        if ai_analysis.get('consensus_count') == 3:  # All 3 AIs agree
            score += 20
        elif ai_analysis.get('consensus_count') == 2:  # 2 out of 3
            score += 10
        
        # Market regime appropriateness (20 points)
        regime = market_data['market_regime']
        action = ai_analysis['action']
        if regime == 'TRENDING' and action in ['buy', 'sell']:
            score += 20  # Good for leverage
        elif regime == 'RANGING':
            score += 10  # Moderate
        elif regime == 'VOLATILE':
            score += 5   # Risky
        
        # Risk/Reward ratio (10 points)
        if ai_analysis.get('risk_reward_ratio', 0) >= 2.0:
            score += 10
        elif ai_analysis.get('risk_reward_ratio', 0) >= 1.5:
            score += 5
        
        # Technical alignment (10 points)
        # All timeframes should agree
        indicators = market_data['indicators']
        timeframe_agreement = 0
        for tf in ['15m', '1h', '4h']:
            if self.is_bullish(indicators[tf]) and action == 'buy':
                timeframe_agreement += 1
            elif self.is_bearish(indicators[tf]) and action == 'sell':
                timeframe_agreement += 1
        
        score += (timeframe_agreement / 3) * 10
        
        return min(score, 100)
    
    async def execute_trade(self, opportunity: dict):
        """
        Execute the trade with proper risk management
        """
        symbol = opportunity['symbol']
        analysis = opportunity['analysis']
        market_data = opportunity['market_data']
        
        # Validate with risk manager
        trade_params = {
            'symbol': symbol,
            'side': analysis['side'],
            'leverage': analysis['suggested_leverage'],
            'stop_loss_percent': analysis['stop_loss_percent'],
            'current_price': market_data['current_price']
        }
        
        validation = await self.risk_manager.validate_trade(trade_params)
        
        if not validation['approved']:
            await self.notifier.send_alert(
                'warning',
                f"Trade rejected by risk manager:\n{validation['reason']}"
            )
            return
        
        # Pass to trade executor
        executor = TradeExecutor()
        await executor.open_position(trade_params, analysis, market_data)
```

### 3. AI Consensus Engine

**Multi-Model Analysis & Agreement:**

```python
"""
AI Consensus Engine
Gets analysis from multiple AI models and requires agreement
"""

class AIConsensusEngine:
    def __init__(self):
        self.claude_client = ClaudeClient()
        self.deepseek_client = DeepSeekClient()
        self.grok_client = GrokClient()  # Optional
        self.cache = AICache()
        
    async def get_consensus(self, symbol: str, market_data: dict) -> dict:
        """
        Get AI consensus - require at least 2 out of 3 models to agree
        """
        
        # Check cache first
        cached = await self.cache.get(symbol, timeframe='5m')
        if cached:
            return cached
        
        # Get analysis from all models in parallel
        analyses = await asyncio.gather(
            self.analyze_with_claude(symbol, market_data),
            self.analyze_with_deepseek(symbol, market_data),
            self.analyze_with_grok(symbol, market_data),
            return_exceptions=True
        )
        
        # Handle any failures gracefully
        valid_analyses = []
        for i, analysis in enumerate(analyses):
            if isinstance(analysis, Exception):
                logger.error(f"AI model {i} failed: {analysis}")
            else:
                valid_analyses.append(analysis)
        
        # Need at least 2 valid analyses
        if len(valid_analyses) < 2:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'consensus': False,
                'reason': 'Insufficient AI responses'
            }
        
        # Count votes
        buy_votes = sum(1 for a in valid_analyses if a['action'] == 'buy')
        sell_votes = sum(1 for a in valid_analyses if a['action'] == 'sell')
        hold_votes = sum(1 for a in valid_analyses if a['action'] == 'hold')
        
        # Determine consensus
        if buy_votes >= 2:
            consensus_action = 'buy'
            consensus_side = 'LONG'
        elif sell_votes >= 2:
            consensus_action = 'sell'
            consensus_side = 'SHORT'
        else:
            consensus_action = 'hold'
            consensus_side = None
        
        # Average confidence from agreeing models
        agreeing_models = [a for a in valid_analyses if a['action'] == consensus_action]
        avg_confidence = sum(a['confidence'] for a in agreeing_models) / len(agreeing_models) if agreeing_models else 0.0
        
        # Build consensus response
        consensus = {
            'action': consensus_action,
            'side': consensus_side,
            'confidence': avg_confidence,
            'consensus': len(agreeing_models) >= 2,
            'consensus_count': len(agreeing_models),
            'suggested_leverage': self.determine_leverage(agreeing_models, market_data),
            'stop_loss_percent': self.determine_stop_loss(agreeing_models),
            'risk_reward_ratio': self.calculate_risk_reward(agreeing_models),
            'reasoning': self.combine_reasoning(agreeing_models),
            'models_used': [a['model_name'] for a in valid_analyses]
        }
        
        # Cache result
        await self.cache.set(symbol, consensus, ttl=300)  # 5 minute cache
        
        return consensus
    
    async def analyze_with_claude(self, symbol: str, market_data: dict) -> dict:
        """
        Get analysis from Claude 4.5 Sonnet
        """
        prompt = self.build_analysis_prompt(symbol, market_data, context='leverage_trading')
        
        response = await self.claude_client.messages.create(
            model="claude-sonnet-4.5-20250929",
            max_tokens=2048,
            temperature=0.3,
            system=LEVERAGE_TRADING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = parse_ai_response(response.content[0].text)
        analysis['model_name'] = 'claude'
        return analysis
    
    async def analyze_with_deepseek(self, symbol: str, market_data: dict) -> dict:
        """
        Get analysis from DeepSeek V3
        """
        # Similar implementation using DeepSeek API
        # ...
        pass
    
    def build_analysis_prompt(self, symbol: str, market_data: dict, context: str) -> str:
        """
        Build comprehensive prompt for AI analysis
        """
        return f"""
You are a professional cryptocurrency leverage trader analyzing {symbol} for a leveraged trade.

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
BB Upper: ${market_data['indicators']['15m']['bb_upper']:.4f}
BB Lower: ${market_data['indicators']['15m']['bb_lower']:.4f}

TECHNICAL INDICATORS (1h timeframe):
RSI: {market_data['indicators']['1h']['rsi']:.1f}
MACD: {market_data['indicators']['1h']['macd']:.4f}

TECHNICAL INDICATORS (4h timeframe):
RSI: {market_data['indicators']['4h']['rsi']:.1f}
MACD: {market_data['indicators']['4h']['macd']:.4f}

Analyze this data and provide your recommendation.

RESPONSE FORMAT (JSON only, no explanations outside JSON):
{{
    "action": "buy" | "sell" | "hold",
    "confidence": 0.0-1.0,
    "side": "LONG" | "SHORT" | null,
    "suggested_leverage": 2-5,
    "stop_loss_percent": 5.0-10.0,
    "entry_price": current_price,
    "stop_loss_price": calculated_price,
    "take_profit_price": calculated_price,
    "risk_reward_ratio": number,
    "reasoning": "brief explanation (max 100 words)",
    "key_factors": ["factor1", "factor2", "factor3"]
}}
"""

# System prompt for leverage trading
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
- Trade in unclear market conditions"""
```

### 4. Position Monitor

**Continuous Position Tracking & Management:**

```python
"""
Position Monitor
Monitors open position every minute
Enforces profit/loss rules and liquidation protection
"""

class PositionMonitor:
    def __init__(self):
        self.exchange = ExchangeClient()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        self.config = TradingConfig()
        
    async def check_position(self, position: dict):
        """
        Main position monitoring function
        Called every 60 seconds when a position is open
        """
        
        symbol = position['symbol']
        
        # Get current price
        ticker = await self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Calculate current P&L
        pnl_data = self.calculate_pnl(position, current_price)
        
        # Update position in database
        await self.update_position_price(position['id'], current_price, pnl_data['unrealized_pnl'])
        
        logger.info(f"Position check: {symbol} | Price: ${current_price:.4f} | P&L: ${pnl_data['unrealized_pnl']:.2f}")
        
        # CRITICAL CHECK 1: Liquidation distance
        if pnl_data['distance_to_liquidation'] < 0.05:  # Less than 5%
            logger.critical(f"LIQUIDATION RISK! Distance: {pnl_data['distance_to_liquidation']*100:.2f}%")
            await self.close_position_immediately(
                position,
                current_price,
                "EMERGENCY - Liquidation risk"
            )
            return
        
        # CRITICAL CHECK 2: Stop-loss hit
        if self.is_stop_loss_hit(position, current_price):
            logger.warning(f"Stop-loss triggered: {symbol}")
            await self.close_position_immediately(
                position,
                current_price,
                "Stop-loss hit"
            )
            return
        
        # CHECK 3: Minimum profit target reached
        if pnl_data['unrealized_pnl'] >= self.config.min_profit_usd:
            # We're in profit above minimum
            # Check if we should take profit now or let it run
            
            # Strategy: If profit is 2x minimum ($5+), close position
            if pnl_data['unrealized_pnl'] >= (self.config.min_profit_usd * 2):
                logger.info(f"Excellent profit achieved: ${pnl_data['unrealized_pnl']:.2f}")
                await self.close_position_immediately(
                    position,
                    current_price,
                    "Take profit - 2x minimum target"
                )
                return
            
            # If price moved significantly beyond min profit price, consider closing
            if position['side'] == 'LONG':
                price_beyond_target = (current_price - position['min_profit_price']) / position['min_profit_price']
                if price_beyond_target > 0.02:  # 2% beyond target
                    await self.close_position_immediately(
                        position,
                        current_price,
                        "Take profit - strong move beyond target"
                    )
                    return
            else:  # SHORT
                price_beyond_target = (position['min_profit_price'] - current_price) / position['min_profit_price']
                if price_beyond_target > 0.02:
                    await self.close_position_immediately(
                        position,
                        current_price,
                        "Take profit - strong move beyond target"
                    )
                    return
        
        # CHECK 4: Get fresh AI opinion (every 5 minutes)
        if self.should_get_fresh_ai_opinion(position):
            market_data = await self.gather_quick_market_data(symbol)
            ai_opinion = await self.get_ai_exit_signal(symbol, market_data, position)
            
            if ai_opinion['should_exit']:
                logger.info(f"AI recommends exit: {ai_opinion['reason']}")
                await self.close_position_immediately(
                    position,
                    current_price,
                    f"AI exit signal - {ai_opinion['reason']}"
                )
                return
        
        # CHECK 5: Send periodic updates
        if self.should_send_update(position):
            await self.notifier.send_portfolio_update(
                await self.get_current_capital(),
                await self.get_daily_pnl(),
                position_with_current_price(position, current_price, pnl_data)
            )
    
    def calculate_pnl(self, position: dict, current_price: float) -> dict:
        """
        Calculate comprehensive P&L metrics
        """
        entry_price = position['entry_price']
        side = position['side']
        leverage = position['leverage']
        position_value = position['position_value_usd']
        
        # Price change percentage
        if side == 'LONG':
            price_change_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            price_change_pct = (entry_price - current_price) / entry_price
        
        # Leveraged P&L
        unrealized_pnl = position_value * price_change_pct * leverage
        
        # Distance to liquidation
        liq_price = position['liquidation_price']
        distance_to_liq = abs(current_price - liq_price) / current_price
        
        # Distance to stop-loss
        sl_price = position['stop_loss_price']
        distance_to_sl = abs(current_price - sl_price) / current_price
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'pnl_percent': price_change_pct * 100,
            'leveraged_pnl_percent': price_change_pct * leverage * 100,
            'distance_to_liquidation': distance_to_liq,
            'distance_to_stop_loss': distance_to_sl
        }
    
    def is_stop_loss_hit(self, position: dict, current_price: float) -> bool:
        """
        Check if stop-loss is triggered
        """
        sl_price = position['stop_loss_price']
        side = position['side']
        
        # Add 0.1% tolerance to avoid false triggers
        if side == 'LONG':
            return current_price <= (sl_price * 1.001)
        else:  # SHORT
            return current_price >= (sl_price * 0.999)
    
    async def close_position_immediately(self, position: dict, current_price: float, reason: str):
        """
        Close position immediately via market order
        """
        symbol = position['symbol']
        side = position['side']
        quantity = position['quantity']
        
        try:
            # Execute close order
            if side == 'LONG':
                order = await self.exchange.create_market_sell_order(symbol, quantity)
            else:  # SHORT
                order = await self.exchange.create_market_buy_order(symbol, quantity)
            
            exit_price = order.get('average', current_price)
            
            # Calculate final P&L
            pnl_data = self.calculate_pnl(position, exit_price)
            realized_pnl = pnl_data['unrealized_pnl']
            
            # Update capital
            await self.update_capital(realized_pnl)
            
            # Record trade history
            trade_duration = (datetime.now() - position['entry_time']).total_seconds()
            await self.record_trade(position, exit_price, realized_pnl, reason, trade_duration)
            
            # Remove active position
            await self.remove_active_position(position['id'])
            
            # Send Telegram notification
            await self.notifier.send_position_closed(
                position,
                realized_pnl,
                reason
            )
            
            # Send updated portfolio
            await self.notifier.send_portfolio_update(
                await self.get_current_capital(),
                await self.get_daily_pnl()
            )
            
            logger.info(f"Position closed: {symbol} | P&L: ${realized_pnl:.2f} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to close position: {e}")
            await self.notifier.send_alert('critical', f"Failed to close position: {e}\nMANUAL INTERVENTION REQUIRED!")
```

### 5. Risk Management System

**Enforces All Safety Rules:**

```python
"""
Risk Management System
Validates all trades against strict rules
Enforces stop-loss, profit targets, circuit breakers
"""

class RiskManager:
    def __init__(self):
        self.config = TradingConfig()
        
    async def validate_trade(self, trade_params: dict) -> dict:
        """
        Comprehensive trade validation
        Returns: {approved: bool, reason: str}
        """
        
        symbol = trade_params['symbol']
        side = trade_params['side']
        leverage = trade_params['leverage']
        stop_loss_percent = trade_params['stop_loss_percent']
        current_price = trade_params['current_price']
        
        # Rule 1: Stop-loss must be between 5-10%
        if stop_loss_percent < 5 or stop_loss_percent > 10:
            return {
                'approved': False,
                'reason': f'Stop-loss {stop_loss_percent}% outside required range (5-10%)'
            }
        
        # Rule 2: Check if we have enough capital
        current_capital = await self.get_current_capital()
        position_value = current_capital * self.config.position_size_percent
        
        if position_value < 10:  # Minimum $10 position
            return {
                'approved': False,
                'reason': f'Insufficient capital: ${current_capital:.2f}'
            }
        
        # Rule 3: Leverage check
        if leverage > self.config.max_leverage:
            return {
                'approved': False,
                'reason': f'Leverage {leverage}x exceeds maximum {self.config.max_leverage}x'
            }
        
        # Rule 4: Daily loss limit check
        daily_pnl = await self.get_daily_pnl()
        max_daily_loss = current_capital * self.config.daily_loss_limit_percent
        
        if daily_pnl < -max_daily_loss:
            return {
                'approved': False,
                'reason': f'Daily loss limit reached: ${daily_pnl:.2f}'
            }
        
        # Rule 5: Consecutive losses check
        consecutive_losses = await self.get_consecutive_losses()
        if consecutive_losses >= self.config.max_consecutive_losses:
            return {
                'approved': False,
                'reason': f'{consecutive_losses} consecutive losses - trading paused'
            }
        
        # Rule 6: Maximum loss per trade check
        max_loss = position_value * (stop_loss_percent / 100) * leverage
        max_acceptable_loss = current_capital * 0.20  # 20% of capital max
        
        if max_loss > max_acceptable_loss:
            return {
                'approved': False,
                'reason': f'Potential loss ${max_loss:.2f} exceeds 20% of capital'
            }
        
        # Rule 7: Liquidation distance must be safe
        if side == 'LONG':
            liq_price = current_price * (1 - (0.9 / leverage))
        else:
            liq_price = current_price * (1 + (0.9 / leverage))
        
        liq_distance = abs(current_price - liq_price) / current_price
        
        if liq_distance < 0.10:  # Must be at least 10% away
            return {
                'approved': False,
                'reason': f'Liquidation too close: {liq_distance*100:.1f}%'
            }
        
        # Rule 8: Minimum profit target validation
        min_profit_pct = self.config.min_profit_usd / position_value
        if min_profit_pct > 0.15:  # More than 15% needed
            return {
                'approved': False,
                'reason': f'Minimum profit target {min_profit_pct*100:.1f}% too high for position size'
            }
        
        # All checks passed
        return {'approved': True, 'reason': 'All risk checks passed'}
```

### 6. Telegram Notifier

**Complete Notification System:**

```python
"""
Telegram Notifier
Sends all real-time updates to user via Telegram
"""

from telegram import Bot
from telegram.constants import ParseMode

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        
    async def send_startup_message(self):
        """Bot started notification"""
        message = """
🤖 <b>AUTONOMOUS TRADING BOT STARTED</b>

The bot is now running in fully autonomous mode.
You will receive notifications for all trading activity.

✅ Auto-scanning for opportunities
✅ AI-powered trade decisions
✅ Automatic position management
✅ Strict risk management enabled

Sit back and monitor your portfolio! 💰
"""
        await self.send_message(message)
    
    async def send_position_opened(self, position: dict):
        """New position opened notification"""
        emoji = "🟢" if position['side'] == 'LONG' else "🔴"
        
        message = f"""
{emoji} <b>NEW POSITION OPENED</b>

💎 <b>{position['symbol']}</b>
📊 Direction: <b>{position['side']} {position['leverage']}x</b>

💰 Position Size: <b>${position['position_value_usd']:.2f}</b>
💵 Entry Price: <b>${position['entry_price']:.4f}</b>
📏 Quantity: <b>{position['quantity']:.6f}</b>

🛑 Stop-Loss: <b>${position['stop_loss_price']:.4f}</b> (-{position['stop_loss_percent']:.1f}%)
💎 Min Profit Target: <b>${position['min_profit_target_usd']:.2f}</b>
⚠️ Liquidation: <b>${position['liquidation_price']:.4f}</b>

🤖 AI Confidence: <b>{position['ai_confidence']*100:.0f}%</b>
🤝 Consensus: <b>{position['ai_model_consensus']}</b>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_position_update(self, position: dict, pnl: float):
        """Periodic position update (every 5 min)"""
        emoji = "💰" if pnl > 0 else "📉"
        
        message = f"""
{emoji} <b>POSITION UPDATE</b>

💎 {position['symbol']} {position['side']} {position['leverage']}x

💵 Entry: ${position['entry_price']:.4f}
💵 Current: ${position['current_price']:.4f}
{emoji} Unrealized P&L: <b>${pnl:+.2f}</b>

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_position_closed(self, position: dict, pnl: float, reason: str):
        """Position closed notification"""
        emoji = "✅" if pnl > 0 else "❌"
        pnl_percent = (pnl / position['position_value_usd']) * 100
        
        message = f"""
{emoji} <b>POSITION CLOSED</b>

💎 <b>{position['symbol']}</b> {position['side']} {position['leverage']}x

💵 Entry: ${position['entry_price']:.4f}
💵 Exit: ${position['exit_price']:.4f}

{emoji} <b>Profit/Loss: ${pnl:+.2f} ({pnl_percent:+.1f}%)</b>

📝 Reason: {reason}
⏰ Duration: {position['duration']}

Current Capital: ${await self.get_current_capital():.2f}
"""
        await self.send_message(message)
    
    async def send_portfolio_update(self, capital: float, daily_pnl: float, position: dict = None):
        """Full portfolio status"""
        message = f"""
💼 <b>PORTFOLIO STATUS</b>

💰 Total Capital: <b>${capital:.2f}</b>
📊 Today's P&L: <b>${daily_pnl:+.2f}</b>

"""
        if position:
            pnl = position.get('unrealized_pnl', 0)
            emoji = "💰" if pnl > 0 else "📉"
            message += f"""
📍 <b>OPEN POSITION:</b>
💎 {position['symbol']} {position['side']} {position['leverage']}x
💵 Entry: ${position['entry_price']:.4f}
💵 Current: ${position['current_price']:.4f}
{emoji} Unrealized: ${pnl:+.2f}
"""
        else:
            message += """
📍 <b>No Open Position</b>
🔍 Scanning for opportunities...
"""
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        await self.send_message(message)
    
    async def send_scan_result(self, symbol: str, confidence: float, action: str):
        """Market scan result"""
        message = f"""
🔍 <b>SCAN COMPLETE</b>

💎 Best Opportunity: <b>{symbol}</b>
🎯 Signal: <b>{action.upper()}</b>
🤖 AI Confidence: <b>{confidence*100:.0f}%</b>

{'📈 Initiating trade...' if confidence >= 0.80 else '⏳ Waiting for stronger signal...'}
"""
        await self.send_message(message)
    
    async def send_alert(self, alert_type: str, message_text: str):
        """General alerts"""
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'success': '✅'
        }
        emoji = emoji_map.get(alert_type, 'ℹ️')
        
        message = f"{emoji} <b>{alert_type.upper()}</b>\n\n{message_text}"
        await self.send_message(message)
    
    async def send_daily_summary(self, summary_data: dict):
        """End of day summary"""
        message = f"""
📊 <b>DAILY SUMMARY</b>
{summary_data['date']}

💰 Starting Capital: ${summary_data['starting_capital']:.2f}
💰 Ending Capital: ${summary_data['ending_capital']:.2f}
{'💹' if summary_data['daily_pnl'] > 0 else '📉'} <b>Daily P&L: ${summary_data['daily_pnl']:+.2f}</b>

📈 Total Trades: {summary_data['total_trades']}
✅ Winners: {summary_data['winning_trades']}
❌ Losers: {summary_data['losing_trades']}
📊 Win Rate: {summary_data['win_rate']:.1f}%

💎 Best Trade: ${summary_data['largest_win']:.2f}
📉 Worst Trade: ${summary_data['largest_loss']:.2f}

See you tomorrow! 🌙
"""
        await self.send_message(message)
    
    async def send_message(self, text: str, parse_mode=ParseMode.HTML):
        """Send message to Telegram"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
```

---

## Environment Configuration

### `.env` File Structure

```env
# Exchange API Credentials
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_here

# AI Model API Keys
CLAUDE_API_KEY=sk-ant-your-claude-key-here
DEEPSEEK_API_KEY=your_deepseek_key_here
GROK_API_KEY=your_grok_key_here  # Optional

# Telegram Configuration
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379/0

# Trading Configuration (Optional - overrides defaults)
INITIAL_CAPITAL=100.00
MAX_LEVERAGE=5
POSITION_SIZE_PERCENT=0.80
MIN_STOP_LOSS_PERCENT=0.05
MAX_STOP_LOSS_PERCENT=0.10
MIN_PROFIT_USD=2.50
MIN_AI_CONFIDENCE=0.75
SCAN_INTERVAL_SECONDS=300
POSITION_CHECK_SECONDS=60

# Risk Management
DAILY_LOSS_LIMIT_PERCENT=0.10
MAX_CONSECUTIVE_LOSSES=3

# Feature Flags
USE_PAPER_TRADING=false
ENABLE_GROK=false  # Set true if you have Grok API access
```

---

## Deployment Instructions

### Docker Deployment (Recommended)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the bot
CMD ["python", "leverage_telegram_bot.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  trading-bot:
    build: .
    container_name: autonomous-trading-bot
    restart: unless-stopped
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - trading-network

  postgres:
    image: postgres:15-alpine
    container_name: trading-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: trading_bot
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - trading-network

  redis:
    image: redis:7-alpine
    container_name: trading-cache
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - trading-network

networks:
  trading-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
```

**Start the bot:**
```bash
# First time setup
docker-compose up -d postgres redis
docker-compose run --rm trading-bot python setup_database.py

# Start the bot
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop the bot
docker-compose down
```

---

## Testing Strategy

### Phase 1: Paper Trading (MANDATORY - 2-4 weeks)

```yaml
Goal: Validate bot logic without risking real money

Setup:
  - Set USE_PAPER_TRADING=true in .env
  - Start with virtual $100
  - Bot operates normally but doesn't execute real trades
  - Simulates trades and tracks performance

Success Criteria:
  - Bot runs continuously for 2+ weeks without crashes
  - Win rate > 50%
  - Average profit per trade > $1.00
  - No critical bugs or logic errors
  - Telegram notifications working correctly

If paper trading is NOT profitable:
  - DO NOT proceed to real money
  - Analyze losing trades
  - Adjust AI prompts or risk parameters
  - Continue paper trading until profitable
```

### Phase 2: Micro-Capital Test ($50-100, 2 weeks)

```yaml
Goal: Test with small real money to verify everything works

Setup:
  - Start with $50-100 real capital
  - Use 2x-3x leverage maximum
  - Very conservative settings

Monitor:
  - All trades executed correctly
  - Stop-losses trigger as expected
  - Telegram notifications accurate
  - No exchange API errors

Success Criteria:
  - Capital preserved or grows
  - System stable
  - No critical issues

If successful → Proceed to Phase 3
If losses > 20% → Return to paper trading
```

### Phase 3: Full Operation ($200-500+)

```yaml
Goal: Normal trading with adequate capital

Setup:
  - Increase capital to $200-500
  - Use 3x-5x leverage (as configured)
  - Continue monitoring closely for first month

Ongoing:
  - Review daily summaries
  - Analyze winning/losing trades
  - Fine-tune AI prompts based on performance
  - Adjust risk parameters if needed
```

---

## Cost Analysis

### Monthly Operating Costs

```yaml
For $100-300 Capital (Conservative Trading):

AI API Costs:
  Scenario: 15-20 analyses per day
  - Claude: 10 analyses/day × $0.15 = $1.50/day = $45/month
  - DeepSeek: 20 analyses/day × $0.02 = $0.40/day = $12/month
  - Grok (optional): 5 analyses/day × $0.10 = $0.50/day = $15/month
  
  With intelligent caching (reduces by 60%):
  - Total AI: $30-40/month ✅

Infrastructure:
  - DigitalOcean Droplet (2GB): $12/month
  - OR Raspberry Pi 4: $0/month (after $60 one-time purchase)
  - Database (included in droplet)
  - Redis (included in droplet)
  
  Total Infrastructure: $12/month (or $0 with Raspberry Pi)

Telegram:
  - Free ✅

TOTAL MONTHLY COST: $42-52/month

Cost/Capital Ratio:
  - $100 capital: 42-52% (high, but acceptable for learning)
  - $300 capital: 14-17% (reasonable)
  - $500 capital: 8-10% (good)

Recommendation: Start with minimum $200-300 capital
```

### Cost Optimization Strategies

```yaml
1. Intelligent AI Usage:
   - Use DeepSeek for routine scans (80% of calls)
   - Use Claude only for trade entry/exit decisions (20%)
   - Cache analysis results for 5 minutes
   - Expected savings: 60-70%

2. Scanning Frequency:
   - No position: Every 10-15 minutes (vs 5)
   - Reduces daily scans by 50%
   - Expected savings: $15-20/month

3. Position Monitoring:
   - Every 2 minutes instead of 1 minute
   - Only call AI every 5 minutes (not every check)
   - Expected savings: $10-15/month

Optimized Total: $25-35/month for $200-300 capital ✅
```

---

## Risk Warnings & Disclaimers

### CRITICAL USER WARNINGS

```
🚨 PLEASE READ CAREFULLY 🚨

LEVERAGE TRADING IS EXTREMELY RISKY:
❌ You can lose 100% of your capital in minutes
❌ 95% of leverage traders lose money
❌ AI cannot predict the future
❌ Markets are unpredictable and volatile

THIS BOT:
✅ Follows strict risk management rules
✅ Uses AI for analysis
✅ Automates trading decisions

BUT IT CANNOT:
❌ Guarantee profits
❌ Prevent losses
❌ Predict market crashes
❌ Protect against black swan events

BEFORE USING WITH REAL MONEY:
1. ✅ Complete 2-4 weeks of paper trading
2. ✅ Verify bot is profitable in simulation
3. ✅ Start with money you can afford to lose
4. ✅ Begin with $50-100 maximum
5. ✅ Use 2x-3x leverage only initially
6. ✅ Monitor daily for first 2 weeks

NEVER:
❌ Invest money you need for living expenses
❌ Use borrowed money
❌ Ignore daily summaries
❌ Disable safety features
❌ Use high leverage (10x+) with small capital

YOU ACCEPT FULL RESPONSIBILITY:
- This is educational software
- No profit guarantees
- Past performance ≠ future results
- You may lose all your money
- Developer is not liable for losses

Type "I UNDERSTAND THE RISKS" to proceed.
```

---

## Success Metrics & KPIs

### Track These Metrics

```yaml
Daily:
  - Total trades executed
  - Win rate (%)
  - Average profit per winning trade
  - Average loss per losing trade
  - Largest single win/loss
  - Daily P&L
  - Circuit breaker activations

Weekly:
  - Capital growth/decline (%)
  - Total P&L