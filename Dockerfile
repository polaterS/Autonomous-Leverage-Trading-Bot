# 🎯 LEVEL-BASED TRADING v5.0.11 - Reliable Exit Price
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V5011_RELIABLE_EXIT_PRICE
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.11: RELIABLE EXIT PRICE FIX!" && \
    echo "   🛡️ CRITICAL: Fix wrong PnL calculation on position close" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.11 FEATURES:" && \
    echo "      ✅ Reliable exit price detection" && \
    echo "      ✅ Fresh ticker fallback if order avg is None" && \
    echo "      ✅ Better precision for low-price coins" && \
    echo "      ✅ Fixed: Profit target hit but $0 PnL bug" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 Previous versions:" && \
    echo "      ✅ v5.0.10: Async Portfolio Updates" && \
    echo "      ✅ v5.0.9: Trendline Price Position Validation" && \
    echo "      ✅ v5.0.8: Level Proximity Re-Validation"

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

# 🎯 CACHE BUST MARKER: v5.0.11 - Reliable Exit Price
# Current deployment: 20251205_V5011_RELIABLE_EXIT_PRICE
# Changes: Fix wrong PnL calculation when position closes
#   🛡️ v5.0.11: Reliable Exit Price
#      ✅ Check close_order.get('average') AND 'price'
#      ✅ If None/0, fetch fresh ticker from exchange
#      ✅ Better precision (.8f) for low-price coins
#      ✅ Fixed: DENT profit target hit but $0 PnL
#   📊 Previous versions:
#      ✅ v5.0.10: Async Portfolio Updates
#      ✅ v5.0.9: Trendline Price Position Validation
#      ✅ v5.0.8: Level Proximity Re-Validation
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
