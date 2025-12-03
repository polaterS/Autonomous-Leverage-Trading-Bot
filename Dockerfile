# 🛡️ PA-ONLY v4.7.6 - ORDER FLOW FIX (non-blocking validation)
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251203_V476_ORDERFLOW_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.6: ORDER FLOW NON-BLOCKING!" && \
    echo "   🔧 Critical Fix:" && \
    echo "      ✅ Order flow validation now WARNING only (not rejection)" && \
    echo "      ✅ Skips order flow check if order book unavailable" && \
    echo "      ✅ Added INFO-level logging for order book diagnosis" && \
    echo "      ✅ Relaxed thresholds: 5% → 2%" && \
    echo "   🛡️ Previous fixes included:" && \
    echo "      ✅ v4.7.5: Global logger for indicators.py" && \
    echo "      ✅ v4.7.4: Order flow weighted_imbalance in returns" && \
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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.6
# Current deployment: 20251203_V476_ORDERFLOW_FIX
# Changes: Order flow validation now NON-BLOCKING
#   🔧 v4.7.6 Critical Fix:
#      ✅ Order flow is now WARNING only (won't block trades)
#      ✅ Skips validation if order book data unavailable
#      ✅ INFO-level logging to diagnose order book fetch issues
#      ✅ Relaxed thresholds: 5% → 2%
#   🛡️ Previous fixes included:
#      ✅ v4.7.5: Global logger for indicators.py
#      ✅ v4.7.4: Order flow weighted_imbalance in returns
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
