"""
🎯 #5: GARCH Volatility Predictor
Forward-looking volatility forecasting using GARCH(1,1) models

Instead of backward-looking ATR, this predicts FUTURE volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("⚠️ arch library not available - GARCH predictions disabled")


class VolatilityPredictor:
    """
    🎯 #5: GARCH-based forward-looking volatility predictor.

    Uses GARCH(1,1) model to forecast next-period volatility.
    More sophisticated than simple historical ATR.
    """

    def __init__(self):
        self.garch_cache: Dict[str, Dict] = {}  # symbol -> {model, last_update, forecast}
        self.cache_ttl_minutes: int = 60  # Update GARCH every hour
        self.min_observations: int = 100  # Need 100+ returns for reliable GARCH

        logger.info("✅ Volatility Predictor initialized (GARCH-based forecasting)")

    def predict_volatility(
        self,
        symbol: str,
        returns: np.ndarray,
        horizon: int = 1
    ) -> Optional[float]:
        """
        🎯 #5: Predict future volatility using GARCH(1,1).

        Args:
            symbol: Trading symbol
            returns: Array of percentage returns (e.g., [0.01, -0.02, 0.015, ...])
            horizon: Forecast horizon in periods (default 1 = next period)

        Returns:
            Predicted volatility as annualized percentage (e.g., 45.2 for 45.2%)
            None if insufficient data or GARCH fails
        """
        if not ARCH_AVAILABLE:
            logger.debug("GARCH unavailable, using fallback")
            return self._fallback_volatility_estimate(returns)

        # Check cache first
        cached = self._get_cached_forecast(symbol)
        if cached is not None:
            return cached

        # Need enough data for GARCH
        if len(returns) < self.min_observations:
            logger.debug(f"Insufficient data for GARCH: {len(returns)} < {self.min_observations}")
            return self._fallback_volatility_estimate(returns)

        try:
            # Scale returns to percentage (GARCH works better with 0-100 scale)
            returns_pct = returns * 100

            # Remove any NaN or inf values
            returns_pct = returns_pct[np.isfinite(returns_pct)]

            if len(returns_pct) < self.min_observations:
                return self._fallback_volatility_estimate(returns)

            # Fit GARCH(1,1) model
            # GARCH(1,1) is the most common specification:
            # σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
            garch_model = arch_model(
                returns_pct,
                vol='Garch',
                p=1,  # GARCH lag order
                q=1,  # ARCH lag order
                rescale=False
            )

            # Fit model (suppress output)
            fitted = garch_model.fit(disp='off', show_warning=False)

            # Forecast next period volatility
            forecast = fitted.forecast(horizon=horizon)

            # Extract conditional variance forecast
            # forecast.variance contains the predicted variance for next period
            predicted_variance = forecast.variance.values[-1, 0]

            # Convert variance to annualized volatility percentage
            # sqrt(variance) = daily vol, * sqrt(365) = annual vol
            predicted_vol_daily = np.sqrt(predicted_variance)
            predicted_vol_annual_pct = predicted_vol_daily * np.sqrt(365)

            # Cache result
            self.garch_cache[symbol] = {
                'forecast': predicted_vol_annual_pct,
                'last_update': datetime.now(),
                'model_params': {
                    'omega': float(fitted.params['omega']),
                    'alpha': float(fitted.params['alpha[1]']),
                    'beta': float(fitted.params['beta[1]'])
                }
            }

            logger.info(
                f"📈 GARCH({symbol}): Predicted vol = {predicted_vol_annual_pct:.1f}% annually | "
                f"α={fitted.params['alpha[1]']:.3f}, β={fitted.params['beta[1]']:.3f}"
            )

            return predicted_vol_annual_pct

        except Exception as e:
            logger.warning(f"GARCH fitting failed for {symbol}: {e}, using fallback")
            return self._fallback_volatility_estimate(returns)

    def _get_cached_forecast(self, symbol: str) -> Optional[float]:
        """Check if we have a recent GARCH forecast cached."""
        if symbol not in self.garch_cache:
            return None

        cached = self.garch_cache[symbol]
        age_minutes = (datetime.now() - cached['last_update']).total_seconds() / 60

        if age_minutes < self.cache_ttl_minutes:
            logger.debug(f"Using cached GARCH forecast for {symbol} (age: {age_minutes:.1f}m)")
            return cached['forecast']

        return None

    def _fallback_volatility_estimate(self, returns: np.ndarray) -> float:
        """
        Fallback to simple historical volatility if GARCH unavailable.

        Returns annualized volatility percentage.
        """
        if len(returns) < 20:
            return 50.0  # Default to 50% annual vol for crypto

        # Calculate standard deviation of returns
        vol_daily = np.std(returns)

        # Annualize: daily vol * sqrt(365)
        vol_annual_pct = vol_daily * np.sqrt(365) * 100

        return vol_annual_pct

    def get_volatility_regime(self, predicted_vol_pct: float) -> str:
        """
        🎯 #5: Classify volatility regime based on predicted volatility.

        Args:
            predicted_vol_pct: Predicted annualized volatility percentage

        Returns:
            Regime string: 'LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'EXTREME_VOL'
        """
        # Thresholds based on crypto market norms
        if predicted_vol_pct < 30:
            return 'LOW_VOL'
        elif predicted_vol_pct < 60:
            return 'NORMAL_VOL'
        elif predicted_vol_pct < 90:
            return 'HIGH_VOL'
        else:
            return 'EXTREME_VOL'

    def calculate_optimal_leverage(
        self,
        predicted_vol_pct: float,
        max_leverage: int = 5
    ) -> int:
        """
        🎯 #5: Calculate optimal leverage based on predicted volatility.

        Higher predicted volatility = lower leverage (safety)

        Args:
            predicted_vol_pct: Predicted annualized volatility percentage
            max_leverage: Maximum allowed leverage

        Returns:
            Recommended leverage (2-max_leverage)
        """
        # Inverse relationship: high vol = low leverage
        if predicted_vol_pct >= 90:
            # EXTREME volatility: minimum leverage
            optimal = 2
        elif predicted_vol_pct >= 60:
            # HIGH volatility: 2-3x
            optimal = 2
        elif predicted_vol_pct >= 40:
            # NORMAL-HIGH volatility: 3x
            optimal = 3
        elif predicted_vol_pct >= 30:
            # NORMAL volatility: 4x
            optimal = 4
        else:
            # LOW volatility: max leverage (stable conditions)
            optimal = max_leverage

        logger.debug(f"Optimal leverage for {predicted_vol_pct:.1f}% vol: {optimal}x")

        return min(optimal, max_leverage)

    def predict_next_day_range(
        self,
        current_price: float,
        predicted_vol_pct: float
    ) -> Tuple[float, float]:
        """
        🎯 #5: Predict likely price range for next day.

        Uses predicted volatility to calculate 68% confidence interval
        (±1 standard deviation).

        Args:
            current_price: Current market price
            predicted_vol_pct: Predicted annualized volatility percentage

        Returns:
            Tuple of (lower_bound, upper_bound) for next day
        """
        # Convert annual vol to daily vol
        vol_daily_pct = predicted_vol_pct / np.sqrt(365)

        # 1 standard deviation = 68% confidence interval
        lower = current_price * (1 - vol_daily_pct / 100)
        upper = current_price * (1 + vol_daily_pct / 100)

        return (lower, upper)


# Singleton instance
_volatility_predictor: Optional[VolatilityPredictor] = None


def get_volatility_predictor() -> VolatilityPredictor:
    """Get or create volatility predictor instance."""
    global _volatility_predictor
    if _volatility_predictor is None:
        _volatility_predictor = VolatilityPredictor()
    return _volatility_predictor
