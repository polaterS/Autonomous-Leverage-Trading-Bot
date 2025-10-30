"""
Bot Controller - Controls the trading bot process.

Production-ready features:
- Process health monitoring
- Automatic crash recovery
- Resource usage tracking
- Graceful shutdown with safety checks
- Comprehensive logging
- Uptime and metrics history
"""

import subprocess
import psutil
import os
import signal
import logging
import time
from typing import Optional, Callable, Dict, List
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BotMetrics:
    """Bot process metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    running: bool


@dataclass
class BotLifecycle:
    """Bot lifecycle tracking."""
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    restart_count: int = 0
    crash_count: int = 0
    total_uptime_seconds: float = 0.0
    last_health_check: Optional[datetime] = None
    metrics_history: List[BotMetrics] = field(default_factory=list)


class BotController:
    """
    Production-ready bot process controller.

    Features:
    - Health monitoring with configurable intervals
    - Automatic crash detection and recovery
    - Resource usage tracking and alerts
    - Graceful shutdown with position safety
    - Comprehensive metrics and logging
    """

    def __init__(self, auto_restart: bool = False, max_restarts: int = 5, use_docker: bool = True):
        """
        Initialize bot controller.

        Args:
            auto_restart: Automatically restart on crashes
            max_restarts: Maximum consecutive restarts before giving up
            use_docker: Use Docker container for bot (default: True)
        """
        self.bot_process: Optional[subprocess.Popen] = None
        self.bot_script_path = Path(__file__).parent.parent.parent / "main.py"
        self.project_root = Path(__file__).parent.parent.parent
        self.use_docker = use_docker
        self.docker_container_name = "autonomous-trading-bot"
        self.lifecycle = BotLifecycle()
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self._status_callbacks: List[Callable] = []
        self._monitoring = False

    # ==================== Core lifecycle methods ====================

    def is_running(self) -> bool:
        """Check if bot is currently running."""
        if self.use_docker:
            # Check if Docker container is running
            try:
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", self.docker_container_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0 and result.stdout.strip() == "true"
            except Exception as e:
                logger.error(f"Error checking Docker container status: {e}")
                return False
        else:
            # Check local process
            if self.bot_process is None:
                return False
            return self.bot_process.poll() is None

    def start_bot(self, validate_first: bool = True) -> bool:
        """
        Start the trading bot with validation.

        Args:
            validate_first: Validate environment before starting

        Returns:
            True if started successfully
        """
        if self.is_running():
            logger.warning("Bot is already running")
            return False

        try:
            if self.use_docker:
                # Start Docker container
                logger.info(f"Starting Docker container: {self.docker_container_name}")
                result = subprocess.run(
                    ["docker-compose", "up", "-d", "trading-bot"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    logger.error(f"Failed to start Docker container: {result.stderr}")
                    return False

                # Wait for container to be running
                time.sleep(2)

                if not self.is_running():
                    logger.error("Container started but not running")
                    return False

            else:
                # Pre-start validation for local bot
                if validate_first and not self._validate_startup():
                    logger.error("Startup validation failed")
                    return False

                # Start bot as subprocess
                self.bot_process = subprocess.Popen(
                    ["python", str(self.bot_script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )

            # Update lifecycle
            self.lifecycle.start_time = datetime.now()
            self.lifecycle.stop_time = None

            logger.info("Bot started successfully")
            self._notify_status_change("started")

            return True

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False

    def stop_bot(self, force: bool = False, check_position: bool = True) -> bool:
        """
        Stop the trading bot gracefully.

        Args:
            force: Force kill without graceful shutdown
            check_position: Check for active positions before stopping

        Returns:
            True if stopped successfully
        """
        if not self.is_running():
            logger.warning("Bot is not running")
            return False

        try:
            # Safety check for active positions (only if not force)
            if check_position and not force:
                if self._has_active_position():
                    logger.warning("Active position detected - use force=True to override")
                    return False

            # Calculate uptime before stopping
            if self.lifecycle.start_time:
                uptime = (datetime.now() - self.lifecycle.start_time).total_seconds()
                self.lifecycle.total_uptime_seconds += uptime

            if self.use_docker:
                # Stop Docker container
                logger.info(f"Stopping Docker container: {self.docker_container_name}")
                result = subprocess.run(
                    ["docker-compose", "stop", "trading-bot"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    logger.error(f"Failed to stop Docker container: {result.stderr}")
                    return False

            else:
                # Stop local process
                if force:
                    logger.warning("Force killing bot process")
                    self.bot_process.kill()
                else:
                    logger.info("Gracefully stopping bot")
                    if os.name == 'nt':  # Windows
                        os.kill(self.bot_process.pid, signal.CTRL_BREAK_EVENT)
                    else:  # Unix
                        self.bot_process.terminate()

                # Wait for process to exit
                self.bot_process.wait(timeout=30)
                self.bot_process = None

            # Update lifecycle
            self.lifecycle.stop_time = datetime.now()

            logger.info("Bot stopped successfully")
            self._notify_status_change("stopped")
            return True

        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timed out, force stopping")
            if self.use_docker:
                subprocess.run(
                    ["docker-compose", "kill", "trading-bot"],
                    cwd=str(self.project_root),
                    timeout=10
                )
            else:
                self.bot_process.kill()
                self.bot_process.wait(timeout=5)
                self.bot_process = None
            self.lifecycle.stop_time = datetime.now()
            logger.info("Bot force stopped after timeout")
            return True

        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return False

    def restart_bot(self, delay: int = 2) -> bool:
        """
        Restart the trading bot.

        Args:
            delay: Seconds to wait between stop and start

        Returns:
            True if restarted successfully
        """
        logger.info(f"Restarting bot (delay: {delay}s)")

        if self.stop_bot():
            time.sleep(delay)

            if self.start_bot():
                self.lifecycle.restart_count += 1
                logger.info(f"Bot restarted successfully (count: {self.lifecycle.restart_count})")
                return True

        logger.error("Failed to restart bot")
        return False

    # ==================== Status and monitoring ====================

    def get_bot_pid(self) -> Optional[int]:
        """Get bot process ID."""
        if self.use_docker and self.is_running():
            try:
                # Get container PID
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Pid}}", self.docker_container_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return int(result.stdout.strip())
            except:
                pass
        elif self.bot_process:
            return self.bot_process.pid
        return None

    def get_bot_status(self) -> Dict:
        """Get comprehensive bot status with metrics."""
        if not self.is_running():
            return {
                'running': False,
                'pid': None,
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'uptime_seconds': 0.0,
                'uptime_str': '0s'
            }

        try:
            if self.use_docker:
                # Get Docker container stats
                result = subprocess.run(
                    ["docker", "stats", self.docker_container_name, "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    stats = result.stdout.strip().split(',')
                    cpu_percent = float(stats[0].rstrip('%'))
                    # Parse memory (e.g., "123.4MiB / 2GiB")
                    mem_parts = stats[1].split('/')
                    mem_str = mem_parts[0].strip()
                    if 'MiB' in mem_str:
                        memory_mb = float(mem_str.replace('MiB', ''))
                    elif 'GiB' in mem_str:
                        memory_mb = float(mem_str.replace('GiB', '')) * 1024
                    else:
                        memory_mb = 0.0
                else:
                    cpu_percent = 0.0
                    memory_mb = 0.0

                pid = self.get_bot_pid()
            else:
                # Get local process stats
                process = psutil.Process(self.bot_process.pid)
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_mb = process.memory_info().rss / 1024 / 1024
                pid = self.bot_process.pid

            # Calculate uptime
            uptime_seconds = 0.0
            uptime_str = '0s'
            if self.lifecycle.start_time:
                uptime_seconds = (datetime.now() - self.lifecycle.start_time).total_seconds()
                uptime_str = self._format_duration(uptime_seconds)

            # Store metrics
            self._record_metrics(cpu_percent, memory_mb)

            return {
                'running': True,
                'pid': pid,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'uptime_seconds': uptime_seconds,
                'uptime_str': uptime_str,
                'restart_count': self.lifecycle.restart_count,
                'crash_count': self.lifecycle.crash_count
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'running': False,
                'pid': None,
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'uptime_seconds': 0.0,
                'uptime_str': '0s'
            }

    def get_health_status(self) -> Dict:
        """
        Get bot health assessment.

        Returns:
            Health status with issues if any
        """
        status = self.get_bot_status()
        health = {
            'healthy': True,
            'issues': [],
            'warnings': []
        }

        if not status['running']:
            health['healthy'] = False
            health['issues'].append("Bot is not running")
            return health

        # Check CPU usage
        if status['cpu_percent'] > 80:
            health['warnings'].append(f"High CPU usage: {status['cpu_percent']:.1f}%")
        if status['cpu_percent'] > 95:
            health['healthy'] = False
            health['issues'].append(f"Critical CPU usage: {status['cpu_percent']:.1f}%")

        # Check memory usage
        if status['memory_mb'] > 500:
            health['warnings'].append(f"High memory usage: {status['memory_mb']:.1f} MB")
        if status['memory_mb'] > 1000:
            health['healthy'] = False
            health['issues'].append(f"Critical memory usage: {status['memory_mb']:.1f} MB")

        # Check crash history
        if self.lifecycle.crash_count > 5:
            health['warnings'].append(f"High crash count: {self.lifecycle.crash_count}")

        return health

    def check_for_crash(self) -> bool:
        """
        Check if bot has crashed unexpectedly.

        Returns:
            True if crash detected
        """
        if not self.is_running() and self.lifecycle.start_time and not self.lifecycle.stop_time:
            # Bot was running but is now stopped without explicit stop
            self.lifecycle.crash_count += 1
            self.lifecycle.stop_time = datetime.now()
            logger.error(f"Bot crash detected (total crashes: {self.lifecycle.crash_count})")

            # Attempt auto-restart if enabled
            if self.auto_restart and self.lifecycle.crash_count < self.max_restarts:
                logger.info(f"Attempting auto-restart ({self.lifecycle.crash_count}/{self.max_restarts})")
                return self.restart_bot(delay=5)

            return True

        return False

    def get_metrics_summary(self, last_n_minutes: int = 60) -> Dict:
        """
        Get summary of metrics history.

        Args:
            last_n_minutes: Time window for analysis

        Returns:
            Metrics summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        recent_metrics = [m for m in self.lifecycle.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                'count': 0,
                'avg_cpu': 0.0,
                'max_cpu': 0.0,
                'avg_memory_mb': 0.0,
                'max_memory_mb': 0.0
            }

        return {
            'count': len(recent_metrics),
            'avg_cpu': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'max_cpu': max(m.cpu_percent for m in recent_metrics),
            'avg_memory_mb': sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            'max_memory_mb': max(m.memory_mb for m in recent_metrics)
        }

    # ==================== Helper methods ====================

    def _validate_startup(self) -> bool:
        """Validate environment before starting bot."""
        # Check if bot script exists
        if not self.bot_script_path.exists():
            logger.error(f"Bot script not found: {self.bot_script_path}")
            return False

        # Check if .env file exists
        env_file = self.bot_script_path.parent / ".env"
        if not env_file.exists():
            logger.error(f".env file not found: {env_file}")
            return False

        logger.info("Startup validation passed")
        return True

    def _has_active_position(self) -> bool:
        """Check if bot has active position (safety check)."""
        try:
            # Import here to avoid circular dependency
            import sys
            sys.path.append(str(Path(__file__).parent))
            from db_controller import DatabaseController

            db = DatabaseController()
            if db.connect():
                position = db.get_active_position()
                db.disconnect()
                return position is not None

        except Exception as e:
            logger.warning(f"Could not check for active position: {e}")

        return False

    def _record_metrics(self, cpu: float, memory_mb: float):
        """Record metrics snapshot."""
        metric = BotMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu,
            memory_mb=memory_mb,
            running=True
        )
        self.lifecycle.metrics_history.append(metric)

        # Keep only last 1000 metrics
        if len(self.lifecycle.metrics_history) > 1000:
            self.lifecycle.metrics_history = self.lifecycle.metrics_history[-1000:]

        self.lifecycle.last_health_check = datetime.now()

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _notify_status_change(self, status: str):
        """Notify registered callbacks of status change."""
        for callback in self._status_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def register_status_callback(self, callback: Callable):
        """Register callback for status changes."""
        self._status_callbacks.append(callback)

    def get_lifecycle_summary(self) -> Dict:
        """Get complete lifecycle summary."""
        return {
            'start_time': self.lifecycle.start_time.isoformat() if self.lifecycle.start_time else None,
            'stop_time': self.lifecycle.stop_time.isoformat() if self.lifecycle.stop_time else None,
            'restart_count': self.lifecycle.restart_count,
            'crash_count': self.lifecycle.crash_count,
            'total_uptime_seconds': self.lifecycle.total_uptime_seconds,
            'total_uptime_str': self._format_duration(self.lifecycle.total_uptime_seconds),
            'metrics_count': len(self.lifecycle.metrics_history)
        }
