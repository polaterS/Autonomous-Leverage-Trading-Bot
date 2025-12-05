# 🎯 LEVEL-BASED TRADING v5.0.0 - S/R Entry System
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V500_LEVEL_BASED_ENTRY
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.0: LEVEL-BASED TRADING SYSTEM!" && \
    echo "   🆕 MAJOR RELEASE: Complete Entry System Redesign" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🎯 NEW PHILOSOPHY:" && \
    echo "      ❌ OLD: Chase trades mid-range, use ML signals" && \
    echo "      ✅ NEW: Wait at S/R levels, require full confirmation" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 NEW FEATURES:" && \
    echo "      ✅ 5 Timeframes: 15m, 1h, 4h, Daily, Weekly" && \
    echo "      ✅ Trend Lines: Ascending/Descending detection" && \
    echo "      ✅ Level Proximity: Only trade within 0.5% of S/R" && \
    echo "      ✅ Triple Confirmation: Candle + Volume 1.5x + RSI" && \
    echo "      ✅ Smart Targets: Next S/R levels as targets" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 ENTRY FLOW:" && \
    echo "      1. Scan 5 timeframes for S/R levels" && \
    echo "      2. Detect trend lines as dynamic S/R" && \
    echo "      3. Check: Price at level? (0.5% proximity)" && \
    echo "      4. At Support → LONG, At Resistance → SHORT" && \
    echo "      5. Require ALL: Candle ✓ Volume ✓ RSI ✓" && \
    echo "      6. If ALL pass → ENTER with tight stop"

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

# 🎯 CACHE BUST MARKER: LEVEL-BASED v5.0.0
# Current deployment: 20251205_V500_LEVEL_BASED_ENTRY
# Changes: Complete entry system redesign - Level-Based Trading
#   🎯 v5.0.0 MAJOR RELEASE:
#      ❌ OLD: Chase mid-range prices with ML signals
#      ✅ NEW: Wait at S/R levels with triple confirmation
#   🆕 NEW FEATURES:
#      ✅ 5 Timeframes: 15m, 1h, 4h, Daily, Weekly
#      ✅ Trend Line Detection: Ascending/Descending support/resistance
#      ✅ Level Proximity: 0.5% threshold - only trade at levels
#      ✅ Triple Confirmation: Candlestick + Volume 1.5x + RSI extreme
#      ✅ Smart Stop Loss: Beyond the S/R level
#      ✅ Smart Targets: Next S/R levels as profit targets
#   📊 Previous versions:
#      ✅ v4.7.13: ADX filter source fix
#      ✅ v4.7.12: Derivatives filter removed
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
