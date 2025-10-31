"""
Simple chart server to serve interactive HTML charts.
Stores charts in memory and serves via FastAPI endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict
import uuid
from datetime import datetime, timedelta
from src.utils import setup_logging

logger = setup_logging()

# In-memory storage for charts (chart_id -> HTML content)
# Auto-expire after 24 hours
_charts_storage: Dict[str, Dict[str, any]] = {}

# FastAPI app instance
chart_app = FastAPI()


def store_chart(html_content: str, symbol: str) -> str:
    """
    Store chart HTML and return unique chart ID.

    Args:
        html_content: HTML content of the chart
        symbol: Trading symbol

    Returns:
        Unique chart ID
    """
    chart_id = str(uuid.uuid4())[:8]  # Short ID
    _charts_storage[chart_id] = {
        'html': html_content,
        'symbol': symbol,
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(hours=24)
    }

    # Clean up expired charts
    _cleanup_expired_charts()

    logger.info(f"📊 Stored chart {chart_id} for {symbol}")
    return chart_id


def _cleanup_expired_charts():
    """Remove expired charts from storage."""
    now = datetime.now()
    expired = [
        chart_id for chart_id, data in _charts_storage.items()
        if data['expires_at'] < now
    ]
    for chart_id in expired:
        del _charts_storage[chart_id]
        logger.debug(f"🗑️ Removed expired chart {chart_id}")


@chart_app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chart Server API",
        "endpoints": {
            "/chart/{chart_id}": "Get interactive chart",
            "/charts": "List all charts"
        },
        "active_charts": len(_charts_storage)
    }


@chart_app.get("/chart/{chart_id}", response_class=HTMLResponse)
async def get_chart(chart_id: str):
    """
    Serve interactive chart HTML.

    Args:
        chart_id: Unique chart identifier

    Returns:
        HTML content of the chart
    """
    if chart_id not in _charts_storage:
        raise HTTPException(status_code=404, detail="Chart not found or expired")

    chart_data = _charts_storage[chart_id]

    # Check if expired
    if chart_data['expires_at'] < datetime.now():
        del _charts_storage[chart_id]
        raise HTTPException(status_code=410, detail="Chart expired")

    logger.info(f"📊 Serving chart {chart_id} ({chart_data['symbol']})")
    return HTMLResponse(content=chart_data['html'])


@chart_app.get("/charts")
async def list_charts():
    """List all active charts."""
    _cleanup_expired_charts()
    return {
        "total_charts": len(_charts_storage),
        "charts": [
            {
                "id": chart_id,
                "symbol": data['symbol'],
                "created_at": data['created_at'].isoformat(),
                "expires_at": data['expires_at'].isoformat()
            }
            for chart_id, data in _charts_storage.items()
        ]
    }


# Singleton chart app
_chart_app_instance = None


def get_chart_app() -> FastAPI:
    """Get FastAPI chart app instance."""
    global _chart_app_instance
    if _chart_app_instance is None:
        _chart_app_instance = chart_app
    return _chart_app_instance
