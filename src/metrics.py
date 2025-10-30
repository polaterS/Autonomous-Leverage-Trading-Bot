"""
Prometheus Metrics for Trading Bot Monitoring.

Exposes key metrics for monitoring and alerting:
- Trade metrics (total, win rate, P&L)
- Position metrics (active, P&L, duration)
- AI metrics (latency, confidence, model performance)
- Exchange metrics (API latency, errors)
- Risk metrics (drawdown, exposure, circuit breakers)

Usage:
    from src.metrics import metrics

    # Record a trade
    metrics.trades_total.labels(side='LONG', result='win').inc()
    metrics.trade_pnl_usd.observe(5.25)

    # Record AI latency
    with metrics.ai_latency.labels(model='qwen').time():
        result = await ai_model.analyze(...)

    # Expose metrics endpoint
    from prometheus_client import start_http_server
    start_http_server(8001)  # Metrics at http://localhost:8001/metrics
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from typing import Dict, Any
from decimal import Decimal


class TradingBotMetrics:
    """Centralized metrics collection for trading bot."""

    def __init__(self):
        # ==================== TRADE METRICS ====================

        self.trades_total = Counter(
            'trading_bot_trades_total',
            'Total number of trades executed',
            ['side', 'result', 'symbol']  # LONG/SHORT, win/loss, BTC/ETH
        )

        self.trade_pnl_usd = Histogram(
            'trading_bot_trade_pnl_usd',
            'Trade P&L in USD',
            buckets=[-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100]
        )

        self.trade_duration_seconds = Histogram(
            'trading_bot_trade_duration_seconds',
            'Trade duration in seconds',
            buckets=[300, 600, 1800, 3600, 7200, 14400, 28800, 86400]  # 5m to 24h
        )

        self.trade_leverage = Histogram(
            'trading_bot_trade_leverage',
            'Leverage used in trades',
            buckets=[1, 2, 3, 5, 7, 10, 15, 20]
        )

        # ==================== POSITION METRICS ====================

        self.active_position_count = Gauge(
            'trading_bot_active_position_count',
            'Number of active positions (0 or 1)'
        )

        self.active_position_pnl_usd = Gauge(
            'trading_bot_active_position_pnl_usd',
            'Current unrealized P&L of active position'
        )

        self.active_position_duration_seconds = Gauge(
            'trading_bot_active_position_duration_seconds',
            'Duration of current active position'
        )

        self.active_position_liquidation_distance_percent = Gauge(
            'trading_bot_active_position_liquidation_distance_percent',
            'Distance to liquidation price (%)'
        )

        # ==================== CAPITAL METRICS ====================

        self.current_capital_usd = Gauge(
            'trading_bot_current_capital_usd',
            'Current total capital'
        )

        self.daily_pnl_usd = Gauge(
            'trading_bot_daily_pnl_usd',
            'Today\'s P&L'
        )

        self.total_pnl_usd = Gauge(
            'trading_bot_total_pnl_usd',
            'All-time P&L'
        )

        self.win_rate_percent = Gauge(
            'trading_bot_win_rate_percent',
            'Overall win rate percentage'
        )

        # ==================== AI METRICS ====================

        self.ai_requests_total = Counter(
            'trading_bot_ai_requests_total',
            'Total AI API requests',
            ['model', 'endpoint']  # qwen/deepseek, entry/exit
        )

        self.ai_latency_seconds = Histogram(
            'trading_bot_ai_latency_seconds',
            'AI API response time',
            ['model'],
            buckets=[0.5, 1, 2, 5, 10, 15, 30, 60]
        )

        self.ai_confidence = Histogram(
            'trading_bot_ai_confidence',
            'AI confidence scores',
            ['model'],
            buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        )

        self.ai_errors_total = Counter(
            'trading_bot_ai_errors_total',
            'AI API errors',
            ['model', 'error_type']
        )

        self.ai_cache_hits_total = Counter(
            'trading_bot_ai_cache_hits_total',
            'AI cache hits',
            ['cache_type']  # redis/postgres
        )

        # ==================== EXCHANGE METRICS ====================

        self.exchange_requests_total = Counter(
            'trading_bot_exchange_requests_total',
            'Total exchange API requests',
            ['endpoint', 'method']
        )

        self.exchange_latency_seconds = Histogram(
            'trading_bot_exchange_latency_seconds',
            'Exchange API response time',
            ['endpoint'],
            buckets=[0.1, 0.2, 0.5, 1, 2, 5, 10]
        )

        self.exchange_errors_total = Counter(
            'trading_bot_exchange_errors_total',
            'Exchange API errors',
            ['error_type']
        )

        self.exchange_rate_limit_remaining = Gauge(
            'trading_bot_exchange_rate_limit_remaining',
            'Remaining rate limit'
        )

        # ==================== RISK METRICS ====================

        self.risk_score = Gauge(
            'trading_bot_risk_score',
            'Current risk score (0-100)'
        )

        self.max_drawdown_percent = Gauge(
            'trading_bot_max_drawdown_percent',
            'Maximum drawdown percentage'
        )

        self.consecutive_losses = Gauge(
            'trading_bot_consecutive_losses',
            'Current consecutive losing trades'
        )

        self.circuit_breakers_triggered_total = Counter(
            'trading_bot_circuit_breakers_triggered_total',
            'Circuit breaker activations',
            ['event_type']  # daily_loss/consecutive_losses/liquidation_risk
        )

        self.slippage_percent = Histogram(
            'trading_bot_slippage_percent',
            'Trade execution slippage',
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
        )

        # ==================== SYSTEM METRICS ====================

        self.bot_uptime_seconds = Gauge(
            'trading_bot_uptime_seconds',
            'Bot uptime in seconds'
        )

        self.bot_restarts_total = Counter(
            'trading_bot_restarts_total',
            'Bot restart count'
        )

        self.database_connections = Gauge(
            'trading_bot_database_connections',
            'Active database connections'
        )

        self.websocket_connections = Gauge(
            'trading_bot_websocket_connections',
            'Active WebSocket connections'
        )

        # ==================== INFO METRICS ====================

        self.bot_info = Info(
            'trading_bot_info',
            'Bot version and configuration'
        )

    # ==================== HELPER METHODS ====================

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        side = trade_data['side']
        symbol = trade_data['symbol']
        pnl = float(trade_data['realized_pnl_usd'])
        result = 'win' if pnl > 0 else 'loss'
        duration = trade_data.get('trade_duration_seconds', 0)
        leverage = trade_data.get('leverage', 1)

        self.trades_total.labels(side=side, result=result, symbol=symbol).inc()
        self.trade_pnl_usd.observe(pnl)
        self.trade_duration_seconds.observe(duration)
        self.trade_leverage.observe(leverage)

    def update_position(self, position: Dict[str, Any]):
        """Update active position metrics."""
        if position:
            self.active_position_count.set(1)
            self.active_position_pnl_usd.set(float(position.get('unrealized_pnl_usd', 0)))

            # Calculate duration
            if 'entry_time' in position:
                from datetime import datetime
                duration = (datetime.now() - position['entry_time']).total_seconds()
                self.active_position_duration_seconds.set(duration)

            # Calculate liquidation distance
            if 'current_price' in position and 'liquidation_price' in position:
                current = float(position['current_price'])
                liq = float(position['liquidation_price'])
                distance = abs(current - liq) / current * 100
                self.active_position_liquidation_distance_percent.set(distance)
        else:
            self.active_position_count.set(0)
            self.active_position_pnl_usd.set(0)
            self.active_position_duration_seconds.set(0)

    def update_capital(self, capital: Decimal, daily_pnl: Decimal, total_pnl: Decimal):
        """Update capital metrics."""
        self.current_capital_usd.set(float(capital))
        self.daily_pnl_usd.set(float(daily_pnl))
        self.total_pnl_usd.set(float(total_pnl))

    def update_win_rate(self, win_rate: float):
        """Update win rate."""
        self.win_rate_percent.set(win_rate * 100)

    def record_ai_request(self, model: str, endpoint: str, latency: float, confidence: float = None):
        """Record AI API request."""
        self.ai_requests_total.labels(model=model, endpoint=endpoint).inc()
        self.ai_latency_seconds.labels(model=model).observe(latency)

        if confidence is not None:
            self.ai_confidence.labels(model=model).observe(confidence)

    def record_ai_error(self, model: str, error_type: str):
        """Record AI error."""
        self.ai_errors_total.labels(model=model, error_type=error_type).inc()

    def record_exchange_request(self, endpoint: str, method: str, latency: float):
        """Record exchange API request."""
        self.exchange_requests_total.labels(endpoint=endpoint, method=method).inc()
        self.exchange_latency_seconds.labels(endpoint=endpoint).observe(latency)

    def record_circuit_breaker(self, event_type: str):
        """Record circuit breaker trigger."""
        self.circuit_breakers_triggered_total.labels(event_type=event_type).inc()

    def set_bot_info(self, version: str, mode: str, symbols: str):
        """Set bot information."""
        self.bot_info.info({
            'version': version,
            'mode': mode,  # 'paper' or 'live'
            'symbols': symbols
        })


# Singleton instance
metrics = TradingBotMetrics()


# Example Grafana dashboard queries:
"""
# Trade win rate (last 24h)
sum(increase(trading_bot_trades_total{result="win"}[24h])) /
sum(increase(trading_bot_trades_total[24h])) * 100

# Average trade P&L
histogram_quantile(0.5, trading_bot_trade_pnl_usd)

# AI latency p95
histogram_quantile(0.95, trading_bot_ai_latency_seconds)

# Active position duration alert (>6 hours)
trading_bot_active_position_duration_seconds > 21600

# Liquidation risk alert (<5%)
trading_bot_active_position_liquidation_distance_percent < 5

# Capital over time
trading_bot_current_capital_usd

# Circuit breaker rate
rate(trading_bot_circuit_breakers_triggered_total[1h])
"""
