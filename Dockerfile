# 🛡️ PA-ONLY v4.7.4 - ORDER FLOW FIX (weighted_imbalance)
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251203_V474_ORDER_FLOW_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.4: ORDER FLOW FIX!" && \
    echo "   🔧 Critical Fix:" && \
    echo "      ✅ weighted_imbalance now returned in all code paths" && \
    echo "      ✅ Order book fetch increased to 100 levels" && \
    echo "      ✅ Debug logging added for troubleshooting" && \
    echo "   🛡️ Previous fixes included:" && \
    echo "      ✅ v4.7.3: market_data passed to validate_trade()" && \
    echo "      ✅ v4.7.2: Trailing Stop, Volume STRICT, Portfolio Risk" && \
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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.4
# Current deployment: 20251203_V474_ORDER_FLOW_FIX
# Changes: Fixed order flow weighted_imbalance always returning 0
#   🔧 v4.7.4 Critical Fix:
#      ✅ indicators.py: Added weighted_imbalance to ALL return paths
#      ✅ market_scanner.py: Order book fetch increased to 100 levels
#      ✅ risk_manager.py: Added debug logging for order flow
#      ✅ This fixes "order flow: 0.0%" always being rejected
#   🛡️ Previous fixes included:
#      ✅ v4.7.3: market_data passed to validate_trade()
#      ✅ v4.7.2: Trailing Stop, Volume STRICT, Portfolio Risk
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
