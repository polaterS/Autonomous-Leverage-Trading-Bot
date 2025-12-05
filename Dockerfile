# 🎯 LEVEL-BASED TRADING v5.0.4 - 2-of-3 Confirmation Fix
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V504B_QUALITY_FILTER_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.4b: QUALITY FILTER FIX!" && \
    echo "   🛡️ CRITICAL FIX: Technical Advanced filter was blocking trades!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.4b FEATURES:" && \
    echo "      ✅ 2-of-3 confirmation rule (was: ALL 3 required)" && \
    echo "      ✅ RSI relaxed: 35/65 (was: 30/70 too strict)" && \
    echo "      ✅ Technical Advanced filter: 10% (was: 40%)" && \
    echo "      ✅ CVD/Ichimoku/Liquidations now optional" && \
    echo "      ✅ Price Action based system prioritized" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🛡️ WHY THIS MATTERS:" && \
    echo "      ❌ v5.0.4: OP/USDT blocked by Tech Advanced 20% < 40%" && \
    echo "      ✅ v5.0.4b: Only filter if Tech Advanced < 10%" && \
    echo "      🎯 Bot will now actually open trades" && \
    echo "      📈 Still high quality - main confluence 60+ required"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 🎯 CACHE BUST MARKER: v5.0.4 - 2-of-3 Confirmation Fix
# Current deployment: 20251205_V504_TWO_OF_THREE_CONFIRMATION
# Changes: Bot was not opening ANY trades due to ALL 3 confirmation requirement
#   🛡️ v5.0.4: 2-of-3 Confirmation Fix
#      ✅ 2-of-3 confirmation rule (was: ALL 3 required - nearly impossible!)
#      ✅ RSI thresholds relaxed: 35/65 (was: 30/70)
#      ✅ Any 2 of (Candlestick, Volume, RSI) = ENTRY ALLOWED
#      ✅ Still high quality - requires 2 confirmations minimum
#   📊 Previous versions:
#      ✅ v5.0.3: Closed Candle + Stop Hunt Detection
#      ✅ v5.0.2: Partial TP + Breakeven System
#      ✅ v5.0.1: /analyze BTC Command
#      ✅ v5.0.0: Level-Based Trading System (complete redesign)
COPY . .

# 🔥 NUCLEAR OPTION: Delete ALL Python cache IMMEDIATELY after copy
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -type f -name "*.pyo" -delete 2>/dev/null || true

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPYCACHEPREFIX=/dev/null

# Expose health check port
EXPOSE 8000

# Health check to ensure bot is running
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the trading bot (FORCE clear ALL Python cache to ensure fresh code)
CMD echo "🔥 DELETING PYTHON CACHE BEFORE STARTUP..." && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -type f -name "*.pyo" -delete 2>/dev/null || true && \
    echo "✅ Cache deletion complete, verifying..." && \
    echo "Remaining .pyc files: $(find /app -type f -name '*.pyc' | wc -l)" && \
    echo "🚀 Starting bot with -B flag (bypass bytecode)..." && \
    python -u -B main.py
