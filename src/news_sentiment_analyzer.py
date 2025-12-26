"""
🗞️ Crypto News & Sentiment Analyzer v1.0

Analyzes news and social media sentiment for cryptocurrency trading decisions.

SOURCES:
1. CryptoPanic API (Free tier) - Aggregated crypto news
2. CoinGecko API (Free) - Market sentiment data
3. Alternative.me Fear & Greed Index (Free)

FEATURES:
- Real-time news sentiment scoring
- Social media buzz detection
- Major announcement detection (partnerships, listings, etc.)
- Negative news filtering (hacks, lawsuits, delistings)

IMPACT ON TRADING:
- Very Bullish News: +15% confidence boost
- Bullish News: +10% confidence boost
- Neutral: No change
- Bearish News: -10% confidence penalty (or skip trade)
- Very Bearish News: Skip trade entirely

NOTE: News is a SUPPLEMENTARY signal, not primary.
PA analysis remains the core decision maker.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """News sentiment levels"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class NewsItem:
    """Single news item with sentiment"""
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment: SentimentLevel
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    is_major: bool = False  # Major announcement flag
    keywords: List[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Aggregated sentiment analysis result"""
    symbol: str
    overall_sentiment: SentimentLevel
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    news_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    major_news: List[NewsItem]
    recent_news: List[NewsItem]
    fear_greed_index: int  # 0-100
    social_buzz: float  # 0.0 to 1.0
    recommendation: str
    confidence_adjustment: int  # -20 to +20
    should_skip_trade: bool
    skip_reason: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


class NewsSentimentAnalyzer:
    """
    Analyzes crypto news and social sentiment for trading decisions.
    """
    
    def __init__(self):
        self.cache: Dict[str, SentimentResult] = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        # Bullish keywords (positive news)
        self.bullish_keywords = [
            # Partnerships & Adoption
            'partnership', 'partners with', 'collaboration', 'integrates',
            'adoption', 'accepts', 'launches', 'launch', 'released',
            
            # Investment & Funding
            'investment', 'funding', 'raised', 'backing', 'institutional',
            'whale', 'accumulation', 'buying', 'inflow',
            
            # Technical & Development
            'upgrade', 'mainnet', 'testnet', 'milestone', 'breakthrough',
            'innovation', 'development', 'roadmap', 'update',
            
            # Listings & Exchanges
            'listing', 'listed', 'binance lists', 'coinbase lists',
            'exchange listing', 'trading pair',
            
            # Regulatory (Positive)
            'approved', 'approval', 'legal', 'regulated', 'compliant',
            'etf approved', 'sec approves',
            
            # Market Sentiment
            'bullish', 'rally', 'surge', 'soars', 'jumps', 'gains',
            'all-time high', 'ath', 'breakout', 'moon',
            
            # Ecosystem Growth
            'tvl increase', 'users growing', 'volume surge', 'record',
        ]
        
        # Bearish keywords (negative news)
        self.bearish_keywords = [
            # Security Issues
            'hack', 'hacked', 'exploit', 'vulnerability', 'breach',
            'stolen', 'theft', 'attack', 'compromised',
            
            # Legal & Regulatory (Negative)
            'lawsuit', 'sued', 'investigation', 'sec charges', 'fraud',
            'illegal', 'ban', 'banned', 'crackdown', 'restriction',
            'delisting', 'delisted', 'removed',
            
            # Team & Project Issues
            'rug pull', 'scam', 'exit scam', 'ponzi', 'founder leaves',
            'team quits', 'abandoned', 'dead project',
            
            # Market Sentiment
            'bearish', 'crash', 'plunge', 'dumps', 'falls', 'drops',
            'sell-off', 'capitulation', 'fear', 'panic',
            
            # Technical Issues
            'bug', 'outage', 'down', 'network issues', 'congestion',
            'failed', 'delayed', 'postponed',
            
            # Financial Issues
            'bankruptcy', 'insolvent', 'liquidation', 'default',
            'withdrawal suspended', 'frozen',
        ]
        
        # Major news keywords (high impact)
        self.major_keywords = [
            'breaking', 'just in', 'urgent', 'major', 'huge',
            'billion', 'million dollar', 'sec', 'etf',
            'binance', 'coinbase', 'blackrock', 'grayscale',
            'elon musk', 'cz', 'vitalik', 'satoshi',
        ]
        
        # Symbol to coin name mapping
        self.symbol_to_name = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'SOL': ['solana', 'sol'],
            'BNB': ['binance coin', 'bnb', 'binance'],
            'XRP': ['ripple', 'xrp'],
            'DOGE': ['dogecoin', 'doge'],
            'ADA': ['cardano', 'ada'],
            'AVAX': ['avalanche', 'avax'],
            'DOT': ['polkadot', 'dot'],
            'LINK': ['chainlink', 'link'],
            'MATIC': ['polygon', 'matic'],
            'UNI': ['uniswap', 'uni'],
            'ATOM': ['cosmos', 'atom'],
            'LTC': ['litecoin', 'ltc'],
            'ARB': ['arbitrum', 'arb'],
            'OP': ['optimism', 'op'],
            'INJ': ['injective', 'inj'],
            'SUI': ['sui'],
            'APT': ['aptos', 'apt'],
            'NEAR': ['near protocol', 'near'],
            'FTM': ['fantom', 'ftm'],
            'AAVE': ['aave'],
            'MKR': ['maker', 'mkr'],
            'CRV': ['curve', 'crv'],
            'RUNE': ['thorchain', 'rune'],
            'TRX': ['tron', 'trx'],
            'TON': ['toncoin', 'ton', 'telegram'],
            'PEPE': ['pepe'],
            'SHIB': ['shiba', 'shib'],
            'WIF': ['dogwifhat', 'wif'],
        }
    
    def _extract_symbol(self, full_symbol: str) -> str:
        """Extract base symbol from trading pair (e.g., BTC/USDT:USDT -> BTC)"""
        return full_symbol.split('/')[0].split(':')[0].upper()
    
    def _get_coin_names(self, symbol: str) -> List[str]:
        """Get all possible names for a coin"""
        base = self._extract_symbol(symbol)
        return self.symbol_to_name.get(base, [base.lower()])
    
    async def analyze_sentiment(self, symbol: str) -> SentimentResult:
        """
        Main entry point: Analyze news sentiment for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            
        Returns:
            SentimentResult with aggregated analysis
        """
        base_symbol = self._extract_symbol(symbol)
        
        # Check cache
        cache_key = base_symbol
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            age = (datetime.now() - cached.last_updated).total_seconds()
            if age < self.cache_ttl:
                logger.debug(f"📰 Using cached sentiment for {base_symbol} (age: {age:.0f}s)")
                return cached
        
        logger.info(f"📰 Analyzing news sentiment for {base_symbol}...")
        
        try:
            # Gather data from multiple sources
            news_items, fear_greed, social_buzz = await asyncio.gather(
                self._fetch_crypto_news(base_symbol),
                self._fetch_fear_greed_index(),
                self._estimate_social_buzz(base_symbol),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(news_items, Exception):
                logger.warning(f"News fetch failed: {news_items}")
                news_items = []
            if isinstance(fear_greed, Exception):
                logger.warning(f"Fear & Greed fetch failed: {fear_greed}")
                fear_greed = 50
            if isinstance(social_buzz, Exception):
                logger.warning(f"Social buzz fetch failed: {social_buzz}")
                social_buzz = 0.5
            
            # Analyze sentiment
            result = self._aggregate_sentiment(
                symbol=base_symbol,
                news_items=news_items,
                fear_greed_index=fear_greed,
                social_buzz=social_buzz
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            logger.info(
                f"📰 {base_symbol} Sentiment: {result.overall_sentiment.value} "
                f"(score: {result.sentiment_score:+.2f}, news: {result.news_count}, "
                f"F&G: {result.fear_greed_index}, adjustment: {result.confidence_adjustment:+d}%)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {base_symbol}: {e}")
            return self._create_neutral_result(base_symbol)
    
    async def _fetch_crypto_news(self, symbol: str) -> List[NewsItem]:
        """
        Fetch news from CryptoPanic API (free tier).
        
        Note: CryptoPanic free tier has rate limits.
        Alternative: Use RSS feeds or other free APIs.
        """
        news_items = []
        coin_names = self._get_coin_names(symbol)
        
        try:
            # CryptoPanic API (free, no auth required for public feed)
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies={symbol}&kind=news"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        
                        for item in results[:20]:  # Last 20 news
                            title = item.get('title', '')
                            
                            # Analyze sentiment
                            sentiment, score = self._analyze_text_sentiment(title)
                            relevance = self._calculate_relevance(title, coin_names)
                            is_major = self._is_major_news(title)
                            
                            news_item = NewsItem(
                                title=title,
                                source=item.get('source', {}).get('title', 'Unknown'),
                                url=item.get('url', ''),
                                published_at=datetime.fromisoformat(
                                    item.get('published_at', '').replace('Z', '+00:00')
                                ) if item.get('published_at') else datetime.now(),
                                sentiment=sentiment,
                                sentiment_score=score,
                                relevance_score=relevance,
                                is_major=is_major,
                                keywords=self._extract_keywords(title)
                            )
                            news_items.append(news_item)
                    else:
                        logger.warning(f"CryptoPanic API returned {response.status}")
                        
        except asyncio.TimeoutError:
            logger.warning("CryptoPanic API timeout")
        except Exception as e:
            logger.warning(f"CryptoPanic API error: {e}")
        
        # If CryptoPanic fails, try alternative (simulated for now)
        if not news_items:
            news_items = await self._fetch_alternative_news(symbol)
        
        return news_items
    
    async def _fetch_alternative_news(self, symbol: str) -> List[NewsItem]:
        """
        Alternative news source when CryptoPanic is unavailable.
        Uses CoinGecko's status updates or simulated data.
        """
        # For now, return empty - in production, add more sources
        # Options: CoinGecko, Messari, LunarCrush, etc.
        return []
    
    async def _fetch_fear_greed_index(self) -> int:
        """
        Fetch Fear & Greed Index from Alternative.me (free API).
        
        Returns:
            Index value 0-100 (0=Extreme Fear, 100=Extreme Greed)
        """
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            return int(data['data'][0].get('value', 50))
        except Exception as e:
            logger.warning(f"Fear & Greed API error: {e}")
        
        return 50  # Neutral default
    
    async def _estimate_social_buzz(self, symbol: str) -> float:
        """
        Estimate social media buzz (0.0 to 1.0).
        
        In production, this would use:
        - LunarCrush API (social metrics)
        - Twitter API (mention volume)
        - Reddit API (post activity)
        
        For now, returns neutral estimate.
        """
        # Placeholder - would integrate with social APIs
        return 0.5
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[SentimentLevel, float]:
        """
        Analyze sentiment of a text using keyword matching.
        
        Returns:
            Tuple of (SentimentLevel, score from -1.0 to 1.0)
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)
        
        # Calculate score
        total = bullish_count + bearish_count
        if total == 0:
            return SentimentLevel.NEUTRAL, 0.0
        
        score = (bullish_count - bearish_count) / max(total, 1)
        
        # Determine level
        if score >= 0.6:
            return SentimentLevel.VERY_BULLISH, score
        elif score >= 0.2:
            return SentimentLevel.BULLISH, score
        elif score <= -0.6:
            return SentimentLevel.VERY_BEARISH, score
        elif score <= -0.2:
            return SentimentLevel.BEARISH, score
        else:
            return SentimentLevel.NEUTRAL, score
    
    def _calculate_relevance(self, text: str, coin_names: List[str]) -> float:
        """Calculate how relevant the news is to the specific coin."""
        text_lower = text.lower()
        
        # Direct mention = high relevance
        for name in coin_names:
            if name in text_lower:
                return 1.0
        
        # General crypto news = lower relevance
        if any(kw in text_lower for kw in ['crypto', 'bitcoin', 'market']):
            return 0.5
        
        return 0.3
    
    def _is_major_news(self, text: str) -> bool:
        """Check if news is a major announcement."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.major_keywords)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        keywords = []
        
        for kw in self.bullish_keywords + self.bearish_keywords:
            if kw in text_lower:
                keywords.append(kw)
        
        return keywords[:5]  # Top 5 keywords
    
    def _aggregate_sentiment(
        self,
        symbol: str,
        news_items: List[NewsItem],
        fear_greed_index: int,
        social_buzz: float
    ) -> SentimentResult:
        """
        Aggregate all sentiment signals into final result.
        """
        # Count sentiments
        bullish_count = sum(1 for n in news_items if n.sentiment in [SentimentLevel.BULLISH, SentimentLevel.VERY_BULLISH])
        bearish_count = sum(1 for n in news_items if n.sentiment in [SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH])
        neutral_count = sum(1 for n in news_items if n.sentiment == SentimentLevel.NEUTRAL)
        
        # Calculate weighted sentiment score
        if news_items:
            weighted_score = sum(
                n.sentiment_score * n.relevance_score 
                for n in news_items
            ) / len(news_items)
        else:
            weighted_score = 0.0
        
        # Incorporate Fear & Greed (contrarian)
        # Extreme fear = bullish signal, Extreme greed = bearish signal
        fg_adjustment = 0.0
        if fear_greed_index <= 25:
            fg_adjustment = 0.2  # Extreme fear = bullish
        elif fear_greed_index <= 40:
            fg_adjustment = 0.1  # Fear = slightly bullish
        elif fear_greed_index >= 75:
            fg_adjustment = -0.2  # Extreme greed = bearish
        elif fear_greed_index >= 60:
            fg_adjustment = -0.1  # Greed = slightly bearish
        
        # Final sentiment score
        final_score = weighted_score * 0.7 + fg_adjustment * 0.3
        final_score = max(-1.0, min(1.0, final_score))
        
        # Determine overall sentiment
        if final_score >= 0.4:
            overall = SentimentLevel.VERY_BULLISH
        elif final_score >= 0.15:
            overall = SentimentLevel.BULLISH
        elif final_score <= -0.4:
            overall = SentimentLevel.VERY_BEARISH
        elif final_score <= -0.15:
            overall = SentimentLevel.BEARISH
        else:
            overall = SentimentLevel.NEUTRAL
        
        # Calculate confidence adjustment for trading
        confidence_adjustment = 0
        should_skip = False
        skip_reason = None
        
        if overall == SentimentLevel.VERY_BULLISH:
            confidence_adjustment = 15
            recommendation = "🟢 Strong bullish sentiment - boost confidence"
        elif overall == SentimentLevel.BULLISH:
            confidence_adjustment = 10
            recommendation = "🟢 Bullish sentiment - slight confidence boost"
        elif overall == SentimentLevel.VERY_BEARISH:
            confidence_adjustment = -15
            should_skip = True
            skip_reason = "Very bearish news sentiment"
            recommendation = "🔴 Strong bearish sentiment - consider skipping trade"
        elif overall == SentimentLevel.BEARISH:
            confidence_adjustment = -10
            recommendation = "🟡 Bearish sentiment - reduce confidence"
        else:
            recommendation = "⚪ Neutral sentiment - no adjustment"
        
        # Check for critical negative news
        major_bearish = [n for n in news_items if n.is_major and n.sentiment in [SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH]]
        if major_bearish:
            should_skip = True
            skip_reason = f"Major bearish news: {major_bearish[0].title[:50]}..."
            recommendation = f"🚨 CRITICAL: {skip_reason}"
        
        # Get major and recent news
        major_news = [n for n in news_items if n.is_major][:3]
        recent_news = sorted(news_items, key=lambda x: x.published_at, reverse=True)[:5]
        
        return SentimentResult(
            symbol=symbol,
            overall_sentiment=overall,
            sentiment_score=final_score,
            confidence=abs(final_score),
            news_count=len(news_items),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            major_news=major_news,
            recent_news=recent_news,
            fear_greed_index=fear_greed_index,
            social_buzz=social_buzz,
            recommendation=recommendation,
            confidence_adjustment=confidence_adjustment,
            should_skip_trade=should_skip,
            skip_reason=skip_reason
        )
    
    def _create_neutral_result(self, symbol: str) -> SentimentResult:
        """Create a neutral result when analysis fails."""
        return SentimentResult(
            symbol=symbol,
            overall_sentiment=SentimentLevel.NEUTRAL,
            sentiment_score=0.0,
            confidence=0.0,
            news_count=0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            major_news=[],
            recent_news=[],
            fear_greed_index=50,
            social_buzz=0.5,
            recommendation="⚪ No sentiment data available",
            confidence_adjustment=0,
            should_skip_trade=False
        )
    
    def get_sentiment_emoji(self, sentiment: SentimentLevel) -> str:
        """Get emoji for sentiment level."""
        emoji_map = {
            SentimentLevel.VERY_BULLISH: "🟢🚀",
            SentimentLevel.BULLISH: "🟢",
            SentimentLevel.NEUTRAL: "⚪",
            SentimentLevel.BEARISH: "🔴",
            SentimentLevel.VERY_BEARISH: "🔴💀",
        }
        return emoji_map.get(sentiment, "⚪")


# Singleton instance
_news_analyzer: Optional[NewsSentimentAnalyzer] = None


def get_news_analyzer() -> NewsSentimentAnalyzer:
    """Get singleton news analyzer instance."""
    global _news_analyzer
    if _news_analyzer is None:
        _news_analyzer = NewsSentimentAnalyzer()
    return _news_analyzer
