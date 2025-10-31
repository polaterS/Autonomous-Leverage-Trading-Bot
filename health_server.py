"""
Comprehensive health check HTTP server.
Provides detailed system status, database health, and trading metrics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime, date
from decimal import Decimal
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Trading Bot Health API", version="1.0.0")

# Import chart server components
from src.chart_server import CHARTS_DIR, _cleanup_expired_charts
from pathlib import Path
import json

startup_time = datetime.now()


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns detailed status of all bot components.
    """
    try:
        from src.database import get_db_client
        from src.config import get_settings

        settings = get_settings()
        uptime_seconds = (datetime.now() - startup_time).total_seconds()

        # Database health check
        db_healthy = False
        db_error = None
        try:
            db = await get_db_client()
            await db.get_trading_config()
            db_healthy = True
        except Exception as e:
            db_error = str(e)

        # Build response
        health_status = {
            "status": "healthy" if db_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime_seconds),
            "uptime_formatted": format_uptime(uptime_seconds),
            "components": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "error": db_error
                },
                "paper_trading": settings.use_paper_trading,
                "ai_models": {
                    "qwen3_max_openrouter": bool(settings.openrouter_api_key),
                    "deepseek_v32": bool(settings.deepseek_api_key)
                }
            }
        }

        if not db_healthy:
            return JSONResponse(status_code=503, content=health_status)

        return health_status

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/status")
async def trading_status():
    """
    Get detailed trading bot status including positions and performance.
    """
    try:
        from src.database import get_db_client

        db = await get_db_client()

        # Get current config
        config = await db.get_trading_config()

        # Get active position
        active_position = await db.get_active_position()

        # Get today's performance
        today_pnl = await db.get_daily_pnl(date.today())

        # Get recent trades
        recent_trades = await db.get_recent_trades(limit=5)

        return {
            "status": "trading" if active_position else "scanning",
            "trading_enabled": config.get('is_trading_enabled', False),
            "capital": {
                "current": float(config.get('current_capital', 0)),
                "initial": float(config.get('initial_capital', 0))
            },
            "daily_pnl": float(today_pnl),
            "active_position": {
                "symbol": active_position['symbol'],
                "side": active_position['side'],
                "leverage": active_position['leverage'],
                "entry_price": float(active_position['entry_price']),
                "current_pnl": float(active_position.get('unrealized_pnl_usd', 0))
            } if active_position else None,
            "recent_trades": len(recent_trades),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get trading performance metrics.
    """
    try:
        from src.database import get_db_client

        db = await get_db_client()

        # Get recent trades for stats
        recent_trades = await db.get_recent_trades(limit=50)

        if not recent_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "message": "No trades executed yet"
            }

        total_trades = len(recent_trades)
        winners = sum(1 for t in recent_trades if t['is_winner'])
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(Decimal(str(t['realized_pnl_usd'])) for t in recent_trades)
        avg_pnl = float(total_pnl / total_trades)

        return {
            "total_trades": total_trades,
            "winning_trades": winners,
            "losing_trades": total_trades - winners,
            "win_rate": round(win_rate, 2),
            "total_pnl": float(total_pnl),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart/{chart_id}", response_class=HTMLResponse)
async def get_chart(chart_id: str):
    """
    Serve interactive chart HTML.

    Args:
        chart_id: Unique chart identifier

    Returns:
        HTML content of the chart
    """
    _cleanup_expired_charts()

    # Check filesystem
    html_file = CHARTS_DIR / f"{chart_id}.html"
    metadata_file = CHARTS_DIR / f"{chart_id}.json"

    if not html_file.exists() or not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Chart not found or expired")

    # Read metadata
    try:
        metadata = json.loads(metadata_file.read_text())
        expires_at = datetime.fromisoformat(metadata['expires_at'])

        # Check if expired
        if expires_at < datetime.now():
            html_file.unlink()
            metadata_file.unlink()
            raise HTTPException(status_code=410, detail="Chart expired")

        # Read and serve HTML
        html_content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/charts")
async def list_charts():
    """List all active charts."""
    _cleanup_expired_charts()

    # List from filesystem
    charts = []
    for metadata_file in CHARTS_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_file.read_text())
            chart_id = metadata_file.stem
            charts.append({
                "id": chart_id,
                "symbol": metadata['symbol'],
                "created_at": metadata['created_at'],
                "expires_at": metadata['expires_at']
            })
        except Exception:
            pass

    return {
        "total_charts": len(charts),
        "charts": charts
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Autonomous Leverage Trading Bot",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/health": "System health check",
            "/status": "Trading bot status",
            "/metrics": "Performance metrics",
            "/chart/{chart_id}": "Get interactive chart",
            "/charts": "List all charts"
        },
        "uptime": format_uptime((datetime.now() - startup_time).total_seconds())
    }


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
