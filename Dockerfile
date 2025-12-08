# 🚫 LOW LIQUIDITY BLACKLIST v5.0.16
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251208_V5016_BLACKLIST
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🚫 v5.0.16: LOW LIQUIDITY COIN BLACKLIST!" && \
    echo "   📊 User request: Block HOT, SLP, ONT, FLOW etc." && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.16 FEATURES:" && \
    echo "      🚫 35+ low liquidity coins BLACKLISTED" && \
    echo "      🚫 HOT, SLP, ONT, FLOW (user identified)" && \
    echo "      🚫 SHIB, PEPE, FLOKI, BONK (meme coins)" && \
    echo "      🚫 LUNC, BTTC, WIN, NFT (very low price)" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 Previous versions:" && \
    echo "      ✅ v5.0.15: Balanced Settings" && \
    echo "      ✅ v5.0.14: Ghost Exit Price Fix"

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

# 🚫 CACHE BUST MARKER: v5.0.16 - Low Liquidity Blacklist
# Current deployment: 20251208_V5016_BLACKLIST
# Changes: User request - block HOT, SLP, ONT, FLOW and similar coins
#   🚫 v5.0.16: Low Liquidity Coin Blacklist
#      🚫 35+ coins blacklisted (unreliable candle patterns)
#      🚫 HOT, SLP, ONT, FLOW (user identified after losses)
#      🚫 SHIB, PEPE, FLOKI, BONK (meme coins)
#      🚫 LUNC, BTTC, WIN, NFT (very low price < $0.01)
#   📊 Previous versions:
#      ✅ v5.0.15: Balanced Settings
#      ✅ v5.0.14: Ghost Exit Price Fix
#      ✅ v5.0.13: Protection Filters
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
