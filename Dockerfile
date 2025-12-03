# 🛡️ PA-ONLY v4.7.2 - CRITICAL RISK MANAGEMENT FIXES
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251203_V472_RISK_MANAGEMENT_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v4.7.2: CRITICAL RISK MANAGEMENT FIXES!" && \
    echo "   🆕 Trailing Stop v2.0:" && \
    echo "      ✅ Min 1% profit BEFORE trailing activates" && \
    echo "      ✅ Prevents premature exits on market noise" && \
    echo "   🆕 Volume Validation STRICT MODE:" && \
    echo "      ✅ Volume data REQUIRED (no bypass)" && \
    echo "      ✅ Min 0.7x average volume threshold" && \
    echo "   🆕 Portfolio Direction Risk:" && \
    echo "      ✅ Max 80% positions same direction" && \
    echo "      ✅ Prevents all-LONG or all-SHORT exposure" && \
    echo "   🆕 Technical Validation STRICT:" && \
    echo "      ✅ market_data REQUIRED (no bypass)" && \
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

# 🛡️ CACHE BUST MARKER: PA-ONLY v4.7.2
# Current deployment: 20251203_V472_RISK_MANAGEMENT_FIX
# Changes: Critical Risk Management Fixes
#   🆕 Trailing Stop v2.0:
#      ✅ Min 1% profit threshold before trailing activates
#      ✅ Prevents premature exits on normal market noise
#      ✅ Position must reach 1% profit before trailing begins
#   🆕 Volume Validation STRICT MODE:
#      ✅ Volume data REQUIRED - cannot be bypassed
#      ✅ Relaxed threshold from 1.2x to 0.7x for low volatility
#      ✅ Prevents low-volume trades (e.g., GALA 0.4x)
#   🆕 Portfolio Direction Risk Check:
#      ✅ Max 80% positions same direction (LONG or SHORT)
#      ✅ Prevents 5/5 LONG or 5/5 SHORT scenarios
#      ✅ Forces diversification, reduces correlation risk
#   🆕 Technical Validation STRICT:
#      ✅ market_data REQUIRED - no bypass allowed
#      ✅ All S/R, volume, order flow checks enforced
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
