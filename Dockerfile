# 🛡️ PA-ONLY v4.7.11 - FILTER REASON DISPLAY FIX
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251204_V4711_FILTER_DISPLAY_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.11: FILTER REASON DISPLAY FIX!" && \
    echo "   🔧 FIXES:" && \
    echo "      ✅ Filter reason now shown in Telegram messages" && \
    echo "      ✅ ATR threshold lowered: 0.5% → 0.3% (less strict)" && \
    echo "   🛡️ ACTIVE FILTERS (5 protection layers):" && \
    echo "      ✅ FILTER 1: Technical Advanced < 40% → Skip trade" && \
    echo "      ✅ FILTER 2: Derivatives = 50% (fallback) → Skip trade" && \
    echo "      ✅ FILTER 3: ATR < 0.3% (low volatility) → Skip trade" && \
    echo "      ✅ FILTER 4: ADX > 40 (trend exhaustion) → Skip trade" && \
    echo "      ✅ FILTER 5: Market 80%+ Neutral → Raise min_score to 70" && \
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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.11
# Current deployment: 20251204_V4711_FILTER_DISPLAY_FIX
# Changes: Fixed filter reason display + adjusted ATR threshold
#   🔧 v4.7.11 FIXES:
#      ✅ Filter reason now shown in Telegram (was showing wrong field)
#      ✅ ATR threshold: 0.5% → 0.3% (0.5% was too strict for neutral market)
#   🛡️ ACTIVE FILTERS (5 protection layers):
#      ✅ FILTER 1: Technical Advanced < 40% → Skip
#      ✅ FILTER 2: Derivatives = 50% (fallback) → Skip
#      ✅ FILTER 3: ATR < 0.3% → Skip (lowered from 0.5%)
#      ✅ FILTER 4: ADX > 40 → Skip
#      ✅ FILTER 5: Market 80%+ Neutral → min_score = 70
#   📊 Previous fixes:
#      ✅ v4.7.10: Quality protection filters
#      ✅ v4.7.9: Confluence scoring fix
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
