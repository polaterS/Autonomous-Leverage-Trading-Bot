# 🛡️ PA-ONLY v4.7.3 - MARKET_DATA FIX FOR STRICT MODE
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251203_V473_MARKET_DATA_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.3: MARKET_DATA FIX FOR STRICT MODE!" && \
    echo "   🔧 Critical Fix:" && \
    echo "      ✅ market_data now passed to validate_trade()" && \
    echo "      ✅ Fixed in: market_scanner, realtime_signal_handler, telegram_bot" && \
    echo "   🛡️ v4.7.2 Features (included):" && \
    echo "      ✅ Trailing Stop v2.0 (1% min profit threshold)" && \
    echo "      ✅ Volume Validation STRICT (0.7x threshold)" && \
    echo "      ✅ Portfolio Direction Risk (max 80% same)" && \
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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.3
# Current deployment: 20251203_V473_MARKET_DATA_FIX
# Changes: Fixed market_data passing to STRICT mode validator
#   🔧 v4.7.3 Critical Fix:
#      ✅ market_data now passed to validate_trade() in all callers
#      ✅ market_scanner.py: Added market_data to trade_params
#      ✅ realtime_signal_handler.py: Added market_data to trade_params
#      ✅ telegram_bot.py: Added market_data to trade_params
#   🛡️ v4.7.2 Features (included):
#      ✅ Trailing Stop v2.0 (1% min profit threshold)
#      ✅ Volume Validation STRICT (0.7x threshold required)
#      ✅ Portfolio Direction Risk (max 80% same direction)
#      ✅ Technical Validation STRICT (market_data required)
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
