# 🚀 PA-ONLY v4.6.0 - INSTITUTIONAL GRADE INDICATORS
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251202_V460_INSTITUTIONAL_INDICATORS
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🏛️ v4.6.0: INSTITUTIONAL GRADE INDICATORS!" && \
    echo "   - Smart Money Concepts (Order Blocks, FVG, Liquidity Sweeps)" && \
    echo "   - Market Structure Analysis (BOS, CHoCH)" && \
    echo "   - Wyckoff Volume Spread Analysis (VSA)" && \
    echo "   - Statistical Edge (Hurst Exponent, Z-Score)" && \
    echo "   - Session Analysis (Asian/European/US)" && \
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

# 🔥 CACHE BUST MARKER: PA-ONLY v4.6.0
# Current deployment: 20251202_V460_INSTITUTIONAL_INDICATORS
# Changes: Added institutional grade indicators for professional analysis
#   - Smart Money Concepts: Order Blocks, Fair Value Gaps, Liquidity Sweeps
#   - Market Structure: Break of Structure (BOS), Change of Character (CHoCH)
#   - Wyckoff VSA: Accumulation/Distribution, Climax, Stopping Volume
#   - Statistical Edge: Hurst Exponent (trend persistence), Z-Score
#   - Session Analysis: Asian/European/US session optimization
#   - 25 points added to confluence scoring for institutional analysis
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
