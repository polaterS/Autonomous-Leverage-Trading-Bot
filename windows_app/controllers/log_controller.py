"""
Log Controller - Reads and monitors bot log files.

Production-ready features:
- Real-time log streaming with callbacks
- Advanced parsing with structured data extraction
- Log level statistics and analytics
- Event detection (trades, errors, warnings)
- Regex-based search
- Log rotation awareness
- Performance metrics extraction
"""

from pathlib import Path
from typing import List, Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re
import logging
from datetime import datetime
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ParsedLog:
    """Structured log entry."""
    timestamp: Optional[datetime]
    level: LogLevel
    component: str
    message: str
    raw: str

    # Extracted data (if applicable)
    symbol: Optional[str] = None
    action: Optional[str] = None
    price: Optional[float] = None
    pnl: Optional[float] = None


@dataclass
class LogStatistics:
    """Log statistics container."""
    total_lines: int = 0
    by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_count: int = 0
    warning_count: int = 0
    trade_entries: int = 0
    trade_exits: int = 0
    last_error: Optional[str] = None
    last_warning: Optional[str] = None


class LogController:
    """
    Production-ready log controller.

    Features:
    - Real-time streaming with callbacks
    - Advanced parsing and event detection
    - Statistics tracking
    - Regex search
    - Performance metrics extraction
    """

    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize log controller.

        Args:
            log_file_path: Path to log file (default: bot.log in project root)
        """
        if log_file_path:
            self.log_file = Path(log_file_path)
        else:
            # Default log file location
            self.log_file = Path(__file__).parent.parent.parent / "bot.log"

        self.last_position = 0
        self.statistics = LogStatistics()
        self._stream_callbacks: List[Callable] = []
        self._event_callbacks: Dict[str, List[Callable]] = defaultdict(list)

    # ==================== Core reading methods ====================

    def get_recent_logs(self, lines: int = 100, parse: bool = False) -> List[Any]:
        """
        Get the most recent log lines.

        Args:
            lines: Number of lines to retrieve
            parse: Return parsed log objects instead of strings

        Returns:
            List of log lines or ParsedLog objects
        """
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return []

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
                recent = [line.strip() for line in all_lines[-lines:]]

                if parse:
                    return [self.parse_log_line_advanced(line) for line in recent]
                return recent

        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []

    def get_new_logs(self, parse: bool = False, notify: bool = True) -> List[Any]:
        """
        Get new log lines since last read (for real-time monitoring).

        Args:
            parse: Return parsed log objects instead of strings
            notify: Trigger callbacks for new logs

        Returns:
            List of new log lines or ParsedLog objects
        """
        if not self.log_file.exists():
            return []

        try:
            # Check for log rotation
            current_size = self.log_file.stat().st_size
            if current_size < self.last_position:
                logger.info("Log rotation detected, resetting position")
                self.last_position = 0

            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()

                new_logs = [line.strip() for line in new_lines if line.strip()]

                # Update statistics
                for log in new_logs:
                    self._update_statistics(log)

                # Notify callbacks
                if notify and new_logs:
                    self._notify_new_logs(new_logs)

                if parse:
                    return [self.parse_log_line_advanced(line) for line in new_logs]
                return new_logs

        except Exception as e:
            logger.error(f"Error reading new logs: {e}")
            return []

    # ==================== Search and filter ====================

    def search_logs(self, keyword: str = None, regex: str = None,
                   max_results: int = 100, case_sensitive: bool = False) -> List[str]:
        """
        Search logs with keyword or regex pattern.

        Args:
            keyword: Simple keyword search (mutually exclusive with regex)
            regex: Regex pattern search
            max_results: Maximum number of results
            case_sensitive: Case sensitive search

        Returns:
            List of matching log lines
        """
        if not self.log_file.exists():
            return []

        if not keyword and not regex:
            logger.warning("No search criteria provided")
            return []

        try:
            matches = []
            pattern = None

            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(regex, flags)

            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_stripped = line.strip()

                    if pattern:
                        if pattern.search(line_stripped):
                            matches.append(line_stripped)
                    elif keyword:
                        compare_line = line_stripped if case_sensitive else line_stripped.lower()
                        compare_keyword = keyword if case_sensitive else keyword.lower()
                        if compare_keyword in compare_line:
                            matches.append(line_stripped)

                    if len(matches) >= max_results:
                        break

            logger.info(f"Search found {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []

    def filter_by_level(self, level: LogLevel, lines: int = 100) -> List[ParsedLog]:
        """
        Filter logs by level.

        Args:
            level: Log level to filter by
            lines: Maximum number of lines to return

        Returns:
            List of filtered ParsedLog objects
        """
        recent_logs = self.get_recent_logs(lines * 3, parse=False)

        filtered = []
        for log in recent_logs:
            parsed = self.parse_log_line_advanced(log)
            if parsed.level == level:
                filtered.append(parsed)
                if len(filtered) >= lines:
                    break

        return filtered

    def get_errors_and_warnings(self, hours: int = 24) -> Dict[str, List[ParsedLog]]:
        """
        Get all errors and warnings from recent hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Dict with 'errors' and 'warnings' lists
        """
        all_logs = self.get_recent_logs(lines=10000, parse=True)

        cutoff = datetime.now().timestamp() - (hours * 3600)
        errors = []
        warnings = []

        for log in all_logs:
            if log.timestamp and log.timestamp.timestamp() >= cutoff:
                if log.level == LogLevel.ERROR or log.level == LogLevel.CRITICAL:
                    errors.append(log)
                elif log.level == LogLevel.WARNING:
                    warnings.append(log)

        return {
            'errors': errors,
            'warnings': warnings
        }

    # ==================== Advanced parsing ====================

    def parse_log_line_advanced(self, log_line: str) -> ParsedLog:
        """
        Parse a log line into structured ParsedLog object with data extraction.

        Args:
            log_line: Raw log line

        Returns:
            ParsedLog object
        """
        # Example log format: 2024-01-15 10:30:45 - trading_bot - INFO - Message here
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([\w_]+) - (\w+) - (.+)'
        match = re.match(pattern, log_line)

        timestamp = None
        component = 'unknown'
        level = LogLevel.INFO
        message = log_line

        if match:
            timestamp_str, component, level_str, message = match.groups()

            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass

            try:
                level = LogLevel(level_str)
            except:
                level = LogLevel.INFO

        # Extract trading data
        symbol = self._extract_symbol(message)
        action = self._extract_action(message)
        price = self._extract_price(message)
        pnl = self._extract_pnl(message)

        return ParsedLog(
            timestamp=timestamp,
            level=level,
            component=component,
            message=message,
            raw=log_line,
            symbol=symbol,
            action=action,
            price=price,
            pnl=pnl
        )

    # Legacy method (for backward compatibility)
    def parse_log_line(self, log_line: str) -> dict:
        """Parse a log line into dict (legacy format)."""
        parsed = self.parse_log_line_advanced(log_line)
        return {
            'timestamp': parsed.timestamp,
            'component': parsed.component,
            'level': parsed.level.value,
            'message': parsed.message,
            'raw': parsed.raw
        }

    # ==================== Statistics and analytics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log statistics."""
        return {
            'total_lines': self.statistics.total_lines,
            'by_level': dict(self.statistics.by_level),
            'error_count': self.statistics.error_count,
            'warning_count': self.statistics.warning_count,
            'trade_entries': self.statistics.trade_entries,
            'trade_exits': self.statistics.trade_exits,
            'last_error': self.statistics.last_error,
            'last_warning': self.statistics.last_warning,
            'error_rate': (self.statistics.error_count / self.statistics.total_lines * 100)
                         if self.statistics.total_lines > 0 else 0
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.statistics = LogStatistics()
        logger.info("Log statistics reset")

    # ==================== Utility methods ====================

    def clear_log_file(self) -> bool:
        """Clear the log file."""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'w') as f:
                    f.write('')
                self.last_position = 0
                self.reset_statistics()
                logger.info("Log file cleared")
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing log file: {e}")
            return False

    def archive_logs(self, archive_path: Optional[Path] = None) -> bool:
        """
        Archive current log file.

        Args:
            archive_path: Path for archive (default: logs/bot_YYYYMMDD_HHMMSS.log)

        Returns:
            True if successful
        """
        if not self.log_file.exists():
            return False

        try:
            if not archive_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_dir = self.log_file.parent / "logs"
                archive_dir.mkdir(exist_ok=True)
                archive_path = archive_dir / f"bot_{timestamp}.log"

            # Copy current log to archive
            import shutil
            shutil.copy2(self.log_file, archive_path)
            logger.info(f"Logs archived to: {archive_path}")

            # Clear current log
            self.clear_log_file()

            return True

        except Exception as e:
            logger.error(f"Error archiving logs: {e}")
            return False

    # ==================== Callbacks and streaming ====================

    def register_stream_callback(self, callback: Callable[[List[str]], None]):
        """Register callback for new log streams."""
        self._stream_callbacks.append(callback)

    def register_event_callback(self, event_type: str, callback: Callable):
        """
        Register callback for specific event types.

        Args:
            event_type: 'trade_entry', 'trade_exit', 'error', 'warning', etc.
            callback: Function to call when event detected
        """
        self._event_callbacks[event_type].append(callback)

    def _notify_new_logs(self, logs: List[str]):
        """Notify stream callbacks of new logs."""
        for callback in self._stream_callbacks:
            try:
                callback(logs)
            except Exception as e:
                logger.error(f"Error in stream callback: {e}")

        # Check for events
        for log in logs:
            self._detect_and_notify_events(log)

    def _detect_and_notify_events(self, log: str):
        """Detect and notify event callbacks."""
        parsed = self.parse_log_line_advanced(log)

        # Detect errors
        if parsed.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            for callback in self._event_callbacks.get('error', []):
                try:
                    callback(parsed)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

        # Detect warnings
        if parsed.level == LogLevel.WARNING:
            for callback in self._event_callbacks.get('warning', []):
                try:
                    callback(parsed)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

        # Detect trade events
        if 'entered' in parsed.message.lower() or 'entry' in parsed.message.lower():
            for callback in self._event_callbacks.get('trade_entry', []):
                try:
                    callback(parsed)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

        if 'exited' in parsed.message.lower() or 'closed' in parsed.message.lower():
            for callback in self._event_callbacks.get('trade_exit', []):
                try:
                    callback(parsed)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

    # ==================== Private helper methods ====================

    def _update_statistics(self, log: str):
        """Update statistics from log line."""
        self.statistics.total_lines += 1

        # Parse to get level
        parsed = self.parse_log_line_advanced(log)
        self.statistics.by_level[parsed.level.value] += 1

        if parsed.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.statistics.error_count += 1
            self.statistics.last_error = parsed.message

        if parsed.level == LogLevel.WARNING:
            self.statistics.warning_count += 1
            self.statistics.last_warning = parsed.message

        # Detect trade events
        if 'entered' in parsed.message.lower():
            self.statistics.trade_entries += 1
        if 'exited' in parsed.message.lower() or 'closed' in parsed.message.lower():
            self.statistics.trade_exits += 1

    def _extract_symbol(self, message: str) -> Optional[str]:
        """Extract trading symbol from message."""
        # Look for patterns like BTCUSDT, ETHUSDT, etc.
        match = re.search(r'\b([A-Z]{3,10}USDT?)\b', message)
        return match.group(1) if match else None

    def _extract_action(self, message: str) -> Optional[str]:
        """Extract trading action from message."""
        message_lower = message.lower()
        if 'buy' in message_lower or 'long' in message_lower or 'entered' in message_lower:
            return 'BUY'
        elif 'sell' in message_lower or 'short' in message_lower:
            return 'SELL'
        elif 'close' in message_lower or 'exit' in message_lower:
            return 'CLOSE'
        return None

    def _extract_price(self, message: str) -> Optional[float]:
        """Extract price from message."""
        # Look for patterns like $12345.67 or price: 12345.67
        patterns = [
            r'\$([0-9,]+\.?[0-9]*)',  # $12,345.67
            r'price[:\s]+([0-9,]+\.?[0-9]*)',  # price: 12345.67
            r'@\s*([0-9,]+\.?[0-9]*)',  # @ 12345.67
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except:
                    pass
        return None

    def _extract_pnl(self, message: str) -> Optional[float]:
        """Extract P&L from message."""
        # Look for patterns like +$123.45, -$123.45, P&L: $123.45
        patterns = [
            r'([+-])\$([0-9,]+\.?[0-9]*)',  # +$123.45
            r'p[&/]?l[:\s]+([+-]?\$?[0-9,]+\.?[0-9]*)',  # P&L: 123.45
            r'profit[:\s]+([+-]?\$?[0-9,]+\.?[0-9]*)',  # profit: 123.45
            r'loss[:\s]+([+-]?\$?[0-9,]+\.?[0-9]*)',  # loss: -123.45
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    # Pattern with separate sign and value
                    sign = match.group(1)
                    value_str = match.group(2).replace(',', '').replace('$', '')
                    try:
                        value = float(value_str)
                        return value if sign == '+' else -value
                    except:
                        pass
                else:
                    # Pattern with combined sign and value
                    value_str = match.group(1).replace(',', '').replace('$', '')
                    try:
                        return float(value_str)
                    except:
                        pass
        return None
