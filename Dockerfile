# 🛡️ PA-ONLY v4.7.8 - THREE PROTECTION FILTERS
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251203_V478_PROTECTION_FILTERS
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.8: THREE CRITICAL PROTECTION FILTERS!" && \
    echo "   🛡️ NEW FILTERS:" && \
    echo "      ✅ ADX > 50 = OVEREXTENDED (no entry)" && \
    echo "      ✅ SIDEWAYS + Volume < 0.7x = NO TRADE" && \
    echo "      ✅ Pullback Detection (23.6%-61.8% Fib)" && \
    echo "   🛡️ Previous fixes:" && \
    echo "      ✅ v4.7.7: Order book methods added" && \
    echo "      ✅ v4.7.6: Order flow non-blocking" && \
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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.8
# Current deployment: 20251203_V478_PROTECTION_FILTERS
# Changes: Added 3 critical protection filters to prevent bad entries
#   🛡️ v4.7.8 PROTECTION FILTERS:
#      ✅ ADX > 50 = Overextended market (no entry, wait for pullback)
#      ✅ SIDEWAYS + Volume < 0.7x = No trading edge (skip)
#      ✅ Pullback Detection (23.6%-61.8% Fib retracement = wait)
#   🛡️ Previous fixes:
#      ✅ v4.7.7: Order book methods (fetch_order_book, fetch_trades)
#      ✅ v4.7.6: Order flow non-blocking validation
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
