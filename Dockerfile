# 🛡️ PA-ONLY v4.7.13 - ADX FILTER SOURCE FIX
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251204_V4713_ADX_SOURCE_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.13: ADX FILTER SOURCE FIX!" && \
    echo "   🔧 CRITICAL FIX:" && \
    echo "      ❌ BUG: ADX filter was using indicators_15m (wrong/missing value)" && \
    echo "      ✅ FIX: Now uses Price Action ADX (same as shown in trade)" && \
    echo "      📝 Example: JASMY ADX 41.7 will now be correctly filtered" && \
    echo "   🛡️ ACTIVE FILTERS (4 protection layers):" && \
    echo "      ✅ FILTER 1: Technical Advanced < 40% → Skip trade" && \
    echo "      ✅ FILTER 2: ATR < 0.3% (low volatility) → Skip trade" && \
    echo "      ✅ FILTER 3: ADX > 40 (from Price Action) → Skip trade" && \
    echo "      ✅ FILTER 4: Market 80%+ Neutral → Raise min_score to 70" && \
    echo "   - Instant Trading still DISABLED"

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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.13
# Current deployment: 20251204_V4713_ADX_SOURCE_FIX
# Changes: Fixed ADX filter to use Price Action ADX (not indicators_15m)
#   🔧 v4.7.13 CRITICAL FIX:
#      ❌ BUG: ADX filter was checking indicators_15m.get('adx', 25)
#      ❌ PROBLEM: indicators_15m had wrong/missing ADX (defaulted to 25)
#      ✅ FIX: Now uses analysis['trend']['adx'] from Price Action
#      ✅ RESULT: ADX 41.7 trades (like JASMY) will now be filtered
#   🛡️ ACTIVE FILTERS (4 protection layers):
#      ✅ FILTER 1: Technical Advanced < 40% → Skip
#      ✅ FILTER 2: ATR < 0.3% → Skip
#      ✅ FILTER 3: ADX > 40 (Price Action) → Skip
#      ✅ FILTER 4: Market 80%+ Neutral → min_score = 70
#   📊 Previous fixes:
#      ✅ v4.7.12: Derivatives filter removed
#      ✅ v4.7.11: Filter reason display fix
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
