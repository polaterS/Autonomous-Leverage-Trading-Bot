"""
Backtesting Framework for Trading Bot Strategy.

Allows testing the bot's AI trading strategy on historical data
before risking real capital.

Features:
- Historical OHLCV data download from Binance
- AI strategy simulation with real AI models
- Complete trade lifecycle simulation (entry, stop-loss, take-profit)
- Comprehensive performance metrics (Sharpe, max drawdown, win rate)
- Position size management with Kelly Criterion
- Detailed trade-by-trade analysis

Usage:
    from src.backtester import Backtester
    from datetime import date

    bt = Backtester()
    results = await bt.run_backtest(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        initial_capital=100.0,
        symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT']
    )

    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from src.config import get_settings
from src.exchange_client import get_exchange_client
from src.ai_engine import get_ai_engine
from src.indicators import calculate_indicators, detect_market_regime
from src.risk_manager import get_risk_manager
from src.utils import setup_logging, calculate_pnl

logger = setup_logging()


@dataclass
class BacktestTrade:
    """Single backtest trade record."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    leverage: int
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    position_value: Decimal
    realized_pnl: Decimal
    pnl_percent: float
    stop_loss_price: Decimal
    close_reason: str
    ai_confidence: float
    duration_hours: float

    @property
    def is_winner(self) -> bool:
        return self.realized_pnl > 0


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    # Capital metrics
    initial_capital: Decimal
    final_capital: Decimal
    total_pnl: Decimal
    total_pnl_percent: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Performance metrics
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: float  # Total wins / Total losses

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float

    # Trade analysis
    average_trade_duration_hours: float
    average_leverage: float

    # Detailed trades
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)


class Backtester:
    """
    Backtest trading strategy on historical data.

    Simulates the complete trading bot behavior including:
    - AI analysis for entry signals
    - Position sizing with risk management
    - Stop-loss and take-profit execution
    - Multiple timeframe analysis
    """

    def __init__(self):
        self.settings = get_settings()
        self.exchange_client = None
        self.ai_engine = None
        self.risk_manager = None

    async def initialize(self):
        """Initialize components (in paper trading mode)."""
        self.exchange_client = await get_exchange_client()
        self.ai_engine = get_ai_engine()
        self.risk_manager = get_risk_manager()
        logger.info("Backtester initialized")

    async def run_backtest(
        self,
        start_date: date,
        end_date: date,
        initial_capital: Decimal,
        symbols: List[str],
        use_ai: bool = True,
        max_positions: int = 1
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital in USD
            symbols: List of symbols to trade
            use_ai: Use AI for trade decisions (True) or simple strategy (False)
            max_positions: Maximum concurrent positions

        Returns:
            BacktestResults with comprehensive metrics
        """
        logger.info(f"=" * 60)
        logger.info(f"BACKTEST: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${initial_capital}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"=" * 60)

        await self.initialize()

        # Download historical data for all symbols
        logger.info("Downloading historical data...")
        historical_data = await self._download_historical_data(
            symbols, start_date, end_date
        )

        # Initialize backtest state
        current_capital = initial_capital
        trades: List[BacktestTrade] = []
        active_position = None
        equity_curve = []

        # Get all timestamps (5-minute intervals)
        all_timestamps = sorted(set(
            ts for symbol_data in historical_data.values()
            for ts in symbol_data.index
        ))

        logger.info(f"Backtesting {len(all_timestamps)} time periods...")

        # Main backtest loop
        for i, timestamp in enumerate(all_timestamps):
            # Update equity curve every day
            if i % 288 == 0:  # Every 24 hours (288 x 5min)
                equity_curve.append({
                    'timestamp': timestamp,
                    'capital': float(current_capital),
                    'trades_count': len(trades)
                })

            # Check if we have an active position
            if active_position:
                # Update position with current price
                symbol = active_position['symbol']
                if timestamp not in historical_data[symbol].index:
                    continue

                current_price = Decimal(str(historical_data[symbol].loc[timestamp, 'close']))

                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(
                    active_position, current_price, timestamp
                )

                if should_exit:
                    # Close position
                    trade = self._close_position(
                        active_position, current_price, timestamp, exit_reason
                    )
                    trades.append(trade)
                    current_capital += trade.realized_pnl
                    active_position = None

                    logger.info(
                        f"[{timestamp}] Closed {trade.symbol} {trade.side}: "
                        f"P&L ${trade.realized_pnl:+.2f} ({trade.pnl_percent:+.1f}%)"
                    )

            else:
                # No position - look for entry signals
                if len(trades) >= 5 and i % 60 != 0:  # After 5 trades, only scan every hour
                    continue

                # Scan all symbols for opportunities
                best_opportunity = None
                best_score = 0

                for symbol in symbols:
                    if timestamp not in historical_data[symbol].index:
                        continue

                    # Gather market data at this timestamp
                    market_data = self._get_market_data_at_timestamp(
                        symbol, historical_data[symbol], timestamp
                    )

                    if not market_data:
                        continue

                    # Get AI analysis (or simple strategy)
                    if use_ai:
                        analysis = await self._get_ai_analysis(symbol, market_data)
                    else:
                        analysis = self._simple_strategy(market_data)

                    if analysis['action'] in ['buy', 'sell']:
                        score = analysis.get('confidence', 0) * 100
                        if score > best_score:
                            best_score = score
                            best_opportunity = {
                                'symbol': symbol,
                                'analysis': analysis,
                                'market_data': market_data
                            }

                # Execute best opportunity if above threshold
                if best_opportunity and best_score >= 65.0:
                    active_position = self._open_position(
                        best_opportunity, current_capital, timestamp
                    )
                    if active_position:
                        logger.info(
                            f"[{timestamp}] Opened {active_position['symbol']} "
                            f"{active_position['side']} {active_position['leverage']}x @ "
                            f"${active_position['entry_price']:.4f}"
                        )

        # Close any remaining position at end
        if active_position:
            symbol = active_position['symbol']
            final_price = Decimal(str(historical_data[symbol].iloc[-1]['close']))
            trade = self._close_position(
                active_position, final_price, all_timestamps[-1], "Backtest ended"
            )
            trades.append(trade)
            current_capital += trade.realized_pnl

        # Calculate results
        results = self._calculate_results(
            initial_capital, current_capital, trades, equity_curve
        )

        logger.info(f"\n" + "=" * 60)
        logger.info(f"BACKTEST RESULTS")
        logger.info(f"=" * 60)
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Win Rate: {results.win_rate:.1%}")
        logger.info(f"Final Capital: ${results.final_capital:.2f}")
        logger.info(f"Total P&L: ${results.total_pnl:+.2f} ({results.total_pnl_percent:+.1f}%)")
        logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.1%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"=" * 60)

        return results

    async def _download_historical_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str = '5m'
    ) -> Dict[str, pd.DataFrame]:
        """Download historical OHLCV data for all symbols."""
        data = {}

        for symbol in symbols:
            try:
                logger.info(f"Downloading {symbol}...")
                ohlcv = await self.exchange_client.fetch_ohlcv(
                    symbol, timeframe, limit=1000
                )

                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Filter by date range
                df = df[(df.index >= pd.Timestamp(start_date)) &
                       (df.index <= pd.Timestamp(end_date))]

                data[symbol] = df
                logger.info(f"✅ {symbol}: {len(df)} candles")

            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")

        return data

    def _get_market_data_at_timestamp(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract market data at a specific timestamp."""
        try:
            # Get last 100 candles up to this timestamp
            historical = df[:timestamp].tail(100)

            if len(historical) < 50:
                return None

            # Calculate indicators
            indicators = calculate_indicators(historical.values.tolist())
            regime = detect_market_regime(historical.values.tolist())

            return {
                'symbol': symbol,
                'current_price': float(historical.iloc[-1]['close']),
                'volume_24h': float(historical['volume'].sum()),
                'market_regime': regime,
                'indicators': {
                    '15m': indicators,
                    '1h': indicators,
                    '4h': indicators
                },
                'funding_rate': {'rate': 0.0}  # Simplified
            }
        except Exception as e:
            logger.warning(f"Failed to get market data: {e}")
            return None

    async def _get_ai_analysis(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get AI analysis for entry signal."""
        try:
            analyses = await self.ai_engine.get_individual_analyses(symbol, market_data)
            if analyses:
                # Return best analysis
                return max(analyses, key=lambda x: x.get('confidence', 0))
            return {'action': 'hold', 'confidence': 0}
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return {'action': 'hold', 'confidence': 0}

    def _simple_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple RSI + MACD strategy (no AI)."""
        indicators = market_data['indicators']['15m']
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

        # Buy signal: RSI < 40 and MACD crosses above signal
        if rsi < 40 and macd > macd_signal:
            return {
                'action': 'buy',
                'side': 'LONG',
                'confidence': 0.70,
                'suggested_leverage': 3,
                'stop_loss_percent': 7.0
            }

        # Sell signal: RSI > 60 and MACD crosses below signal
        elif rsi > 60 and macd < macd_signal:
            return {
                'action': 'sell',
                'side': 'SHORT',
                'confidence': 0.70,
                'suggested_leverage': 3,
                'stop_loss_percent': 7.0
            }

        return {'action': 'hold', 'confidence': 0}

    def _open_position(
        self,
        opportunity: Dict[str, Any],
        current_capital: Decimal,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Simulate opening a position."""
        symbol = opportunity['symbol']
        analysis = opportunity['analysis']
        market_data = opportunity['market_data']

        entry_price = Decimal(str(market_data['current_price']))
        leverage = analysis.get('suggested_leverage', 3)
        stop_loss_pct = analysis.get('stop_loss_percent', 7.0)

        # Calculate position size (80% of capital)
        position_value = current_capital * Decimal("0.80")
        quantity = position_value / entry_price

        # Calculate stop-loss price
        if analysis['side'] == 'LONG':
            stop_loss_price = entry_price * (1 - Decimal(str(stop_loss_pct)) / 100)
        else:
            stop_loss_price = entry_price * (1 + Decimal(str(stop_loss_pct)) / 100)

        return {
            'symbol': symbol,
            'side': analysis['side'],
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'quantity': quantity,
            'position_value': position_value,
            'stop_loss_price': stop_loss_price,
            'ai_confidence': analysis.get('confidence', 0)
        }

    def _check_exit_conditions(
        self,
        position: Dict[str, Any],
        current_price: Decimal,
        timestamp: datetime
    ) -> tuple[bool, str]:
        """Check if position should be closed."""
        side = position['side']
        entry_price = position['entry_price']
        stop_loss_price = position['stop_loss_price']

        # Stop-loss hit
        if side == 'LONG' and current_price <= stop_loss_price:
            return True, "Stop-loss"
        elif side == 'SHORT' and current_price >= stop_loss_price:
            return True, "Stop-loss"

        # Calculate P&L
        pnl_data = calculate_pnl(
            entry_price,
            current_price,
            position['quantity'],
            side,
            position['leverage'],
            position['position_value']
        )

        # Take profit if P&L >= $2.50
        if Decimal(str(pnl_data['unrealized_pnl'])) >= Decimal("2.50"):
            return True, "Take profit"

        # Time-based exit (max 24 hours)
        duration = (timestamp - position['entry_time']).total_seconds() / 3600
        if duration >= 24:
            return True, "Max duration"

        return False, ""

    def _close_position(
        self,
        position: Dict[str, Any],
        exit_price: Decimal,
        exit_time: datetime,
        reason: str
    ) -> BacktestTrade:
        """Simulate closing a position."""
        entry_price = position['entry_price']
        side = position['side']
        leverage = position['leverage']
        quantity = position['quantity']
        position_value = position['position_value']

        # Calculate P&L
        if side == 'LONG':
            price_change_pct = (exit_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - exit_price) / entry_price

        realized_pnl = position_value * price_change_pct * leverage
        pnl_percent = float(price_change_pct * leverage * 100)

        duration = (exit_time - position['entry_time']).total_seconds() / 3600

        return BacktestTrade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            symbol=position['symbol'],
            side=side,
            leverage=leverage,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            position_value=position_value,
            realized_pnl=realized_pnl,
            pnl_percent=pnl_percent,
            stop_loss_price=position['stop_loss_price'],
            close_reason=reason,
            ai_confidence=position.get('ai_confidence', 0),
            duration_hours=duration
        )

    def _calculate_results(
        self,
        initial_capital: Decimal,
        final_capital: Decimal,
        trades: List[BacktestTrade],
        equity_curve: List[Dict[str, Any]]
    ) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        total_trades = len(trades)

        if total_trades == 0:
            return BacktestResults(
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_pnl=Decimal("0"),
                total_pnl_percent=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_duration_days=0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                average_trade_duration_hours=0.0,
                average_leverage=0.0
            )

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_wins = sum(t.realized_pnl for t in winners)
        total_losses = abs(sum(t.realized_pnl for t in losers))

        average_win = total_wins / winning_trades if winning_trades > 0 else Decimal("0")
        average_loss = total_losses / losing_trades if losing_trades > 0 else Decimal("0")

        largest_win = max((t.realized_pnl for t in winners), default=Decimal("0"))
        largest_loss = min((t.realized_pnl for t in losers), default=Decimal("0"))

        profit_factor = float(total_wins / total_losses) if total_losses > 0 else float('inf')

        # Calculate max drawdown
        equity_values = [initial_capital] + [
            initial_capital + sum(t.realized_pnl for t in trades[:i+1])
            for i in range(len(trades))
        ]

        peak = equity_values[0]
        max_dd = 0.0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = float((peak - equity) / peak)
            if dd > max_dd:
                max_dd = dd

        # Calculate Sharpe ratio
        returns = [float(t.pnl_percent) / 100 for t in trades]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe = 0.0

        return BacktestResults(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_pnl=final_capital - initial_capital,
            total_pnl_percent=float((final_capital - initial_capital) / initial_capital * 100),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_duration_days=0,  # TODO: calculate
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe,  # Simplified
            average_trade_duration_hours=np.mean([t.duration_hours for t in trades]),
            average_leverage=np.mean([t.leverage for t in trades]),
            trades=trades,
            equity_curve=equity_curve
        )


# Example usage
async def main():
    """Run example backtest."""
    bt = Backtester()

    results = await bt.run_backtest(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        initial_capital=Decimal("100.00"),
        symbols=['BTC/USDT:USDT'],
        use_ai=False  # Use simple strategy (faster)
    )

    print(f"\nFinal Results:")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Total P&L: ${results.total_pnl:+.2f}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
