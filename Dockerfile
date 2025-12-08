# 🔥 LEVEL-BASED TRADING v5.0.14 - Ghost Exit Price Fix
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251208_V5014_GHOST_EXIT_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🔥 v5.0.14: GHOST EXIT PRICE BUG FIX!" && \
    echo "   🐛 FIX: CELO/DENT Entry=Exit $0 PnL bug!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.14 FEATURES:" && \
    echo "      ✅ Ghost positions now fetch REAL exit price" && \
    echo "      ✅ Proper PnL calculation from Binance trades" && \
    echo "      ✅ + v5.0.13 protection filters included" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 Previous versions:" && \
    echo "      ✅ v5.0.13: Critical Protection Filters" && \
    echo "      ✅ v5.0.12: Bulletproof Exit Price" && \
    echo "      ✅ v5.0.11: Reliable Exit Price"

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

# 🔥 CACHE BUST MARKER: v5.0.14 - Ghost Exit Price Fix
# Current deployment: 20251208_V5014_GHOST_EXIT_FIX
# Changes: CELO/DENT Entry=Exit $0 PnL bug fixed!
#   🔥 v5.0.14: Ghost Exit Price Bug Fix
#      ✅ Ghost positions now fetch REAL exit price from Binance trades
#      ✅ Proper PnL calculation before removing position
#      ✅ Records trade history with correct exit price
#      ✅ + Includes v5.0.13 protection filters
#   📊 Previous versions:
#      ✅ v5.0.13: Critical Protection Filters
#      ✅ v5.0.12: Bulletproof Exit Price
#      ✅ v5.0.11: Reliable Exit Price
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
