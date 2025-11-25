"""
= VOLUME PROFILE ANALYZER - Professional Grade


CRITICAL FOR 75%+ WIN RATE: Price alone is NOT enough for Support/Resistance

Why Volume Profile?
-------------------
Traditional S/R uses only PRICE (swing highs/lows, round numbers, etc.)
BUT: What matters is WHERE THE VOLUME IS.

Key Concepts:
1. VPOC (Volume Point of Control)
   - Price level with MOST volume
   - Acts as MAGNET  Price tends to return to VPOC
   - Strongest support/resistance

2. HVN (High Volume Nodes)
   - Price levels with HEAVY volume
   - Strong support/resistance zones
   - Hard for price to break through

3. LVN (Low Volume Nodes)
   - Price levels with LITTLE volume
   - Weak zones  Price moves FAST through these
   - Good for targets (price doesn't stay here)

4. Value Area (VA)
   - VAH (Value Area High) = Top of 70% volume
   - VAL (Value Area Low) = Bottom of 70% volume
   - Price spends 70% of time in VA

Real Trading Application:
- BUY near VAL/HVN with confluence  Strong support
- SELL near VAH/HVN with confluence  Strong resistance
- AVOID trading at LVN  Price won't stay, no support
- Target LVN zones  Price moves through them quickly


"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from decimal import Decimal
import logging

logger = logging.getLogger('trading_bot')


class VolumeProfileAnalyzer:
    """
    Professional Volume Profile Analysis.

    Identifies where the REAL support and resistance is based on volume,
    not just price patterns.
    """

    def __init__(self, price_bins: int = 50):
        """
        Initialize Volume Profile Analyzer.

        Args:
            price_bins: Number of price levels to divide range into (default 50)
        """
        self.price_bins = price_bins

        # Volume thresholds for classification
        self.hvn_percentile = 75  # Top 25% volume = High Volume Node
        self.lvn_percentile = 25  # Bottom 25% volume = Low Volume Node

        # Clustering tolerance for merging nearby levels
        self.cluster_tolerance_pct = 0.003  # 0.3% - merge levels within this range

        # Minimum volume significance (% of total)
        self.min_volume_significance = 0.02  # 2% of total volume

        logger.info(f"= Volume Profile Analyzer initialized (bins={price_bins})")

    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method: Analyze volume profile and identify key levels.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            Dict containing:
                - vpoc: Volume Point of Control (price)
                - vpoc_volume: Volume at VPOC
                - vah: Value Area High
                - val: Value Area Low
                - hvn_levels: List of High Volume Nodes
                - lvn_levels: List of Low Volume Nodes
                - volume_distribution: Full volume profile
        """
        try:
            if len(df) < 20:
                logger.warning(" Insufficient data for volume profile analysis")
                return self._get_empty_profile()

            # Build volume profile
            volume_profile = self._build_volume_profile(df)

            if not volume_profile:
                return self._get_empty_profile()

            # Calculate key levels
            vpoc_data = self._calculate_vpoc(volume_profile)
            value_area = self._calculate_value_area(volume_profile)
            hvn_levels = self._identify_hvn_levels(volume_profile)
            lvn_levels = self._identify_lvn_levels(volume_profile)

            # Calculate current price distance to key levels
            current_price = float(df['close'].iloc[-1])
            nearest_hvn = self._find_nearest_level(current_price, hvn_levels)
            nearest_lvn = self._find_nearest_level(current_price, lvn_levels)

            result = {
                'vpoc': vpoc_data['price'],
                'vpoc_volume': vpoc_data['volume'],
                'vpoc_distance_pct': ((current_price - vpoc_data['price']) / current_price) * 100,

                'vah': value_area['vah'],
                'val': value_area['val'],
                'value_area_range_pct': value_area['range_pct'],
                'price_in_value_area': value_area['val'] <= current_price <= value_area['vah'],

                'hvn_levels': hvn_levels,
                'hvn_count': len(hvn_levels),
                'nearest_hvn': nearest_hvn,

                'lvn_levels': lvn_levels,
                'lvn_count': len(lvn_levels),
                'nearest_lvn': nearest_lvn,

                'volume_distribution': volume_profile,
                'current_price': current_price
            }

            self._log_volume_profile(result)

            return result

        except Exception as e:
            logger.error(f"L Error analyzing volume profile: {e}", exc_info=True)
            return self._get_empty_profile()

    def _build_volume_profile(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Build volume profile: Distribute volume across price levels.

        Returns:
            List of dicts: [{'price': 50000, 'volume': 1234.5}, ...]
            Sorted by price ascending
        """
        try:
            # Get price range
            price_min = float(df['low'].min())
            price_max = float(df['high'].max())
            price_range = price_max - price_min

            if price_range <= 0:
                return []

            # Create price bins
            bin_size = price_range / self.price_bins
            bins = np.linspace(price_min, price_max, self.price_bins + 1)

            # Initialize volume accumulator for each bin
            volume_per_bin = np.zeros(self.price_bins)

            # Distribute volume across price bins for each candle
            for idx, row in df.iterrows():
                candle_low = float(row['low'])
                candle_high = float(row['high'])
                candle_volume = float(row['volume'])

                # Find which bins this candle touches
                # Distribute volume proportionally across the price range of the candle
                for bin_idx in range(self.price_bins):
                    bin_low = bins[bin_idx]
                    bin_high = bins[bin_idx + 1]
                    bin_mid = (bin_low + bin_high) / 2

                    # Check if this bin is within the candle's price range
                    if bin_mid >= candle_low and bin_mid <= candle_high:
                        # Distribute volume proportionally
                        # Assumption: Volume is evenly distributed within the candle's range
                        overlap = min(bin_high, candle_high) - max(bin_low, candle_low)
                        if overlap > 0:
                            proportion = overlap / (candle_high - candle_low) if candle_high > candle_low else 1.0
                            volume_per_bin[bin_idx] += candle_volume * proportion

            # Build volume profile list
            volume_profile = []
            total_volume = volume_per_bin.sum()

            for bin_idx in range(self.price_bins):
                price_level = (bins[bin_idx] + bins[bin_idx + 1]) / 2
                volume = volume_per_bin[bin_idx]
                volume_pct = (volume / total_volume * 100) if total_volume > 0 else 0

                # Only include levels with significant volume
                if volume_pct >= self.min_volume_significance:
                    volume_profile.append({
                        'price': price_level,
                        'volume': volume,
                        'volume_pct': volume_pct
                    })

            # Sort by price
            volume_profile.sort(key=lambda x: x['price'])

            return volume_profile

        except Exception as e:
            logger.error(f"L Error building volume profile: {e}", exc_info=True)
            return []

    def _calculate_vpoc(self, volume_profile: List[Dict]) -> Dict[str, float]:
        """
        Calculate VPOC (Volume Point of Control).

        Returns:
            Dict with 'price' and 'volume' of the VPOC level
        """
        if not volume_profile:
            return {'price': 0.0, 'volume': 0.0}

        # Find level with maximum volume
        vpoc_level = max(volume_profile, key=lambda x: x['volume'])

        return {
            'price': vpoc_level['price'],
            'volume': vpoc_level['volume']
        }

    def _calculate_value_area(self, volume_profile: List[Dict]) -> Dict[str, float]:
        """
        Calculate Value Area: Price range containing 70% of volume.

        Algorithm:
        1. Start from VPOC
        2. Expand up and down to include 70% of total volume
        """
        if not volume_profile:
            return {'vah': 0.0, 'val': 0.0, 'range_pct': 0.0}

        try:
            total_volume = sum(level['volume'] for level in volume_profile)
            target_volume = total_volume * 0.70  # 70% value area

            # Find VPOC index
            vpoc_level = max(volume_profile, key=lambda x: x['volume'])
            vpoc_idx = next(i for i, level in enumerate(volume_profile) if level['price'] == vpoc_level['price'])

            # Expand from VPOC to include 70% volume
            accumulated_volume = vpoc_level['volume']
            lower_idx = vpoc_idx
            upper_idx = vpoc_idx

            # Expand alternating up and down, choosing the side with more volume
            while accumulated_volume < target_volume:
                # Check if we can expand down
                can_expand_down = lower_idx > 0
                # Check if we can expand up
                can_expand_up = upper_idx < len(volume_profile) - 1

                if not can_expand_down and not can_expand_up:
                    break  # Can't expand further

                # Decide which direction to expand
                if can_expand_down and can_expand_up:
                    # Expand toward side with more volume
                    volume_below = volume_profile[lower_idx - 1]['volume']
                    volume_above = volume_profile[upper_idx + 1]['volume']

                    if volume_below > volume_above:
                        lower_idx -= 1
                        accumulated_volume += volume_profile[lower_idx]['volume']
                    else:
                        upper_idx += 1
                        accumulated_volume += volume_profile[upper_idx]['volume']
                elif can_expand_down:
                    lower_idx -= 1
                    accumulated_volume += volume_profile[lower_idx]['volume']
                elif can_expand_up:
                    upper_idx += 1
                    accumulated_volume += volume_profile[upper_idx]['volume']

            val = volume_profile[lower_idx]['price']
            vah = volume_profile[upper_idx]['price']
            range_pct = ((vah - val) / val * 100) if val > 0 else 0

            return {
                'val': val,
                'vah': vah,
                'range_pct': range_pct
            }

        except Exception as e:
            logger.error(f"L Error calculating value area: {e}", exc_info=True)
            return {'vah': 0.0, 'val': 0.0, 'range_pct': 0.0}

    def _identify_hvn_levels(self, volume_profile: List[Dict]) -> List[Dict[str, float]]:
        """
        Identify High Volume Nodes (HVN): Strong support/resistance.

        Returns:
            List of HVN levels: [{'price': 50000, 'volume': 1234, 'strength': 0.95}, ...]
        """
        if not volume_profile:
            return []

        try:
            # Calculate volume threshold for HVN (top 25% by volume)
            volumes = [level['volume'] for level in volume_profile]
            hvn_threshold = np.percentile(volumes, self.hvn_percentile)

            # Find all levels above threshold
            hvn_candidates = [
                level for level in volume_profile
                if level['volume'] >= hvn_threshold
            ]

            # Cluster nearby HVN levels
            hvn_levels = self._cluster_levels(hvn_candidates)

            # Calculate strength score (0-1) based on volume
            max_volume = max(volumes)
            for hvn in hvn_levels:
                hvn['strength'] = min(hvn['volume'] / max_volume, 1.0)
                hvn['type'] = 'HVN'

            # Sort by volume (strongest first)
            hvn_levels.sort(key=lambda x: x['volume'], reverse=True)

            return hvn_levels

        except Exception as e:
            logger.error(f"L Error identifying HVN levels: {e}", exc_info=True)
            return []

    def _identify_lvn_levels(self, volume_profile: List[Dict]) -> List[Dict[str, float]]:
        """
        Identify Low Volume Nodes (LVN): Weak zones, price moves fast through these.

        Returns:
            List of LVN levels: [{'price': 51000, 'volume': 50, 'weakness': 0.90}, ...]
        """
        if not volume_profile:
            return []

        try:
            # Calculate volume threshold for LVN (bottom 25% by volume)
            volumes = [level['volume'] for level in volume_profile]
            lvn_threshold = np.percentile(volumes, self.lvn_percentile)

            # Find all levels below threshold
            lvn_candidates = [
                level for level in volume_profile
                if level['volume'] <= lvn_threshold
            ]

            # Cluster nearby LVN levels
            lvn_levels = self._cluster_levels(lvn_candidates)

            # Calculate weakness score (0-1) - lower volume = higher weakness
            min_volume = min(volumes) if volumes else 1
            max_volume = max(volumes) if volumes else 1
            for lvn in lvn_levels:
                # Invert: lowest volume = highest weakness
                lvn['weakness'] = 1.0 - ((lvn['volume'] - min_volume) / (max_volume - min_volume) if max_volume > min_volume else 0)
                lvn['type'] = 'LVN'

            # Sort by volume (weakest first)
            lvn_levels.sort(key=lambda x: x['volume'])

            return lvn_levels

        except Exception as e:
            logger.error(f"L Error identifying LVN levels: {e}", exc_info=True)
            return []

    def _cluster_levels(self, levels: List[Dict]) -> List[Dict]:
        """
        Cluster nearby price levels together.

        If two levels are within cluster_tolerance_pct, merge them.
        """
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])

        clustered = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            last_price = current_cluster[-1]['price']
            current_price = level['price']

            # Check if within tolerance
            distance_pct = abs(current_price - last_price) / last_price
            if distance_pct <= self.cluster_tolerance_pct:
                # Add to current cluster
                current_cluster.append(level)
            else:
                # Finish current cluster, start new one
                clustered.append(self._merge_cluster(current_cluster))
                current_cluster = [level]

        # Don't forget the last cluster
        if current_cluster:
            clustered.append(self._merge_cluster(current_cluster))

        return clustered

    def _merge_cluster(self, cluster: List[Dict]) -> Dict:
        """
        Merge a cluster of levels into single representative level.

        Use volume-weighted average price.
        """
        if len(cluster) == 1:
            return cluster[0]

        total_volume = sum(level['volume'] for level in cluster)

        # Volume-weighted average price
        weighted_price = sum(level['price'] * level['volume'] for level in cluster) / total_volume

        return {
            'price': weighted_price,
            'volume': total_volume,
            'volume_pct': sum(level.get('volume_pct', 0) for level in cluster),
            'cluster_size': len(cluster)
        }

    def _find_nearest_level(self, current_price: float, levels: List[Dict]) -> Dict[str, Any]:
        """
        Find nearest level to current price and calculate distance.

        Returns:
            Dict with 'price', 'distance_pct', 'direction' ('above' or 'below')
        """
        if not levels:
            return {'price': None, 'distance_pct': None, 'direction': None}

        # Find closest level
        closest_level = min(levels, key=lambda x: abs(x['price'] - current_price))

        distance_pct = ((closest_level['price'] - current_price) / current_price) * 100
        direction = 'above' if closest_level['price'] > current_price else 'below'

        return {
            'price': closest_level['price'],
            'distance_pct': abs(distance_pct),
            'direction': direction,
            'volume': closest_level['volume']
        }

    def get_support_resistance_from_volume(
        self,
        current_price: float,
        volume_profile_data: Dict,
        max_distance_pct: float = 5.0
    ) -> Dict[str, List[float]]:
        """
        Get actionable support and resistance levels from volume profile.

        Args:
            current_price: Current price
            volume_profile_data: Output from analyze_volume_profile()
            max_distance_pct: Maximum distance to consider (default 5%)

        Returns:
            Dict with 'support_levels' and 'resistance_levels'
            Each level is a dict with 'price' and 'strength' (0-1)
        """
        support_levels = []
        resistance_levels = []

        try:
            # Get HVN levels
            hvn_levels = volume_profile_data.get('hvn_levels', [])

            # Get VPOC
            vpoc = volume_profile_data.get('vpoc', 0)
            vpoc_volume = volume_profile_data.get('vpoc_volume', 0)

            # Add VPOC if within range
            if vpoc > 0:
                distance_pct = abs((vpoc - current_price) / current_price) * 100
                if distance_pct <= max_distance_pct:
                    if vpoc < current_price:
                        support_levels.append({
                            'price': vpoc,
                            'strength': 1.0,  # VPOC is strongest
                            'type': 'VPOC'
                        })
                    else:
                        resistance_levels.append({
                            'price': vpoc,
                            'strength': 1.0,
                            'type': 'VPOC'
                        })

            # Add HVN levels within range
            for hvn in hvn_levels:
                price = hvn['price']
                distance_pct = abs((price - current_price) / current_price) * 100

                if distance_pct <= max_distance_pct:
                    level_data = {
                        'price': price,
                        'strength': hvn.get('strength', 0.8),
                        'type': 'HVN',
                        'volume': hvn['volume']
                    }

                    if price < current_price:
                        support_levels.append(level_data)
                    else:
                        resistance_levels.append(level_data)

            # Sort by proximity to current price
            support_levels.sort(key=lambda x: current_price - x['price'])
            resistance_levels.sort(key=lambda x: x['price'] - current_price)

            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }

        except Exception as e:
            logger.error(f"L Error extracting S/R from volume profile: {e}", exc_info=True)
            return {'support_levels': [], 'resistance_levels': []}

    def _log_volume_profile(self, result: Dict):
        """Log volume profile analysis results."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("= VOLUME PROFILE ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"= VPOC: ${result['vpoc']:.2f} (Volume: {result['vpoc_volume']:.0f})")
        logger.info(f"= Distance to VPOC: {result['vpoc_distance_pct']:+.2f}%")
        logger.info(f"")
        logger.info(f"= Value Area High (VAH): ${result['vah']:.2f}")
        logger.info(f"= Value Area Low (VAL): ${result['val']:.2f}")
        logger.info(f"= VA Range: {result['value_area_range_pct']:.2f}%")
        logger.info(f" Price in VA: {result['price_in_value_area']}")
        logger.info(f"")
        logger.info(f"= High Volume Nodes: {result['hvn_count']}")
        if result['nearest_hvn']['price']:
            logger.info(f"   Nearest: ${result['nearest_hvn']['price']:.2f} ({result['nearest_hvn']['direction']}, {result['nearest_hvn']['distance_pct']:.2f}%)")
        logger.info(f"")
        logger.info(f"=4 Low Volume Nodes: {result['lvn_count']}")
        if result['nearest_lvn']['price']:
            logger.info(f"   Nearest: ${result['nearest_lvn']['price']:.2f} ({result['nearest_lvn']['direction']}, {result['nearest_lvn']['distance_pct']:.2f}%)")
        logger.info("=" * 70)
        logger.info("")

    def _get_empty_profile(self) -> Dict[str, Any]:
        """Return empty profile data."""
        return {
            'vpoc': 0.0,
            'vpoc_volume': 0.0,
            'vpoc_distance_pct': 0.0,
            'vah': 0.0,
            'val': 0.0,
            'value_area_range_pct': 0.0,
            'price_in_value_area': False,
            'hvn_levels': [],
            'hvn_count': 0,
            'nearest_hvn': {'price': None, 'distance_pct': None, 'direction': None},
            'lvn_levels': [],
            'lvn_count': 0,
            'nearest_lvn': {'price': None, 'distance_pct': None, 'direction': None},
            'volume_distribution': [],
            'current_price': 0.0
        }


# Factory function for easy usage
def analyze_volume_profile(df: pd.DataFrame, price_bins: int = 50) -> Dict[str, Any]:
    """
    Convenience function to analyze volume profile.

    Usage:
        volume_data = analyze_volume_profile(df)
        vpoc = volume_data['vpoc']
        hvn_levels = volume_data['hvn_levels']

        # Get actionable S/R levels
        analyzer = VolumeProfileAnalyzer()
        sr_levels = analyzer.get_support_resistance_from_volume(
            current_price=df['close'].iloc[-1],
            volume_profile_data=volume_data
        )
    """
    analyzer = VolumeProfileAnalyzer(price_bins=price_bins)
    return analyzer.analyze_volume_profile(df)
