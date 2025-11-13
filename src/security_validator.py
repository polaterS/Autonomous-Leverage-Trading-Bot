"""
Security Validation & Hardening Module

Provides startup security checks and ongoing security monitoring:
- API permission validation
- Withdrawal permission check (MUST be disabled)
- Rate limiting for external APIs
- Environment variable validation
- Security audit logging
"""

import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time

from src.utils import setup_logging
from src.telegram_notifier import get_notifier

logger = setup_logging()


class RateLimiter:
    """
    Rate limiter for API calls and notifications.

    Prevents:
    - Telegram API ban (max 30 msgs/min)
    - Exchange API rate limit violations
    - AI API quota exhaustion
    """

    def __init__(self):
        self.limits = {
            'telegram': {'max_calls': 30, 'window_seconds': 60},  # 30 per minute
            'exchange': {'max_calls': 1200, 'window_seconds': 60},  # 1200 per minute (Binance limit)
            'ai_api': {'max_calls': 60, 'window_seconds': 60}  # 60 per minute (conservative)
        }

        # Track calls per service
        self.call_history = defaultdict(list)

    def can_call(self, service: str) -> bool:
        """
        Check if can make API call without violating rate limit.

        Args:
            service: 'telegram', 'exchange', or 'ai_api'

        Returns:
            True if within rate limit, False otherwise
        """
        if service not in self.limits:
            logger.warning(f"Unknown service for rate limiting: {service}")
            return True  # Allow by default

        limit_config = self.limits[service]
        max_calls = limit_config['max_calls']
        window_seconds = limit_config['window_seconds']

        # Clean old calls outside window
        now = time.time()
        cutoff_time = now - window_seconds

        self.call_history[service] = [
            call_time for call_time in self.call_history[service]
            if call_time > cutoff_time
        ]

        # Check if under limit
        current_calls = len(self.call_history[service])

        if current_calls >= max_calls:
            logger.warning(
                f"⚠️ Rate limit reached for {service}: "
                f"{current_calls}/{max_calls} calls in last {window_seconds}s"
            )
            return False

        return True

    def record_call(self, service: str):
        """Record an API call for rate limiting."""
        self.call_history[service].append(time.time())

    def get_remaining_calls(self, service: str) -> int:
        """Get remaining calls before hitting rate limit."""
        if service not in self.limits:
            return 999999  # No limit

        limit_config = self.limits[service]
        max_calls = limit_config['max_calls']
        window_seconds = limit_config['window_seconds']

        # Clean old calls
        now = time.time()
        cutoff_time = now - window_seconds

        self.call_history[service] = [
            call_time for call_time in self.call_history[service]
            if call_time > cutoff_time
        ]

        current_calls = len(self.call_history[service])
        remaining = max_calls - current_calls

        return max(0, remaining)


class SecurityValidator:
    """
    Startup security validation and ongoing monitoring.

    Critical checks:
    - Binance withdrawal permissions MUST be disabled
    - Environment variables present
    - Database credentials secure
    - API keys format valid
    """

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.validation_results = {}
        self.last_validation_time = None

    async def run_startup_checks(self) -> Dict[str, Any]:
        """
        Run all security checks at startup.

        Returns:
            Dict with validation results
        """
        logger.info("🔐 Running security validation checks...")

        results = {
            'timestamp': datetime.now().isoformat(),
            'checks_passed': 0,
            'checks_failed': 0,
            'checks_warned': 0,
            'details': {}
        }

        # Check 1: Environment variables
        env_check = self._check_environment_variables()
        results['details']['environment'] = env_check
        self._update_counts(results, env_check['status'])

        # Check 2: API key format validation
        key_check = self._check_api_key_formats()
        results['details']['api_keys'] = key_check
        self._update_counts(results, key_check['status'])

        # Check 3: Binance withdrawal permissions
        binance_check = await self._check_binance_permissions()
        results['details']['binance_permissions'] = binance_check
        self._update_counts(results, binance_check['status'])

        # Check 4: Database security
        db_check = self._check_database_security()
        results['details']['database'] = db_check
        self._update_counts(results, db_check['status'])

        # Check 5: Redis security
        redis_check = self._check_redis_security()
        results['details']['redis'] = redis_check
        self._update_counts(results, redis_check['status'])

        self.validation_results = results
        self.last_validation_time = datetime.now()

        # Log summary
        logger.info(
            f"✅ Security checks complete: "
            f"{results['checks_passed']} passed, "
            f"{results['checks_warned']} warnings, "
            f"{results['checks_failed']} failed"
        )

        # Send Telegram notification
        await self._send_validation_summary(results)

        # Critical failure check
        if results['checks_failed'] > 0:
            logger.critical("🚨 CRITICAL SECURITY ISSUES DETECTED!")
            logger.critical("Review security validation report above")

            # Check for withdrawal permission specifically
            if binance_check['status'] == 'FAILED':
                logger.critical("=" * 60)
                logger.critical("🚨 BINANCE WITHDRAWAL PERMISSIONS ENABLED!")
                logger.critical("THIS IS A CRITICAL SECURITY RISK!")
                logger.critical("Bot will NOT start until this is fixed.")
                logger.critical("=" * 60)
                raise SecurityError("Binance withdrawal permissions must be disabled")

        return results

    def _check_environment_variables(self) -> Dict[str, Any]:
        """Check all required environment variables are set."""
        required_vars = [
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'DATABASE_URL',
            'REDIS_URL'
        ]

        missing = []
        present = []

        for var in required_vars:
            if os.getenv(var):
                present.append(var)
            else:
                missing.append(var)

        if missing:
            return {
                'status': 'FAILED',
                'message': f"Missing environment variables: {', '.join(missing)}",
                'missing_vars': missing
            }

        return {
            'status': 'PASSED',
            'message': f"All {len(required_vars)} required environment variables present",
            'present_vars': present
        }

    def _check_api_key_formats(self) -> Dict[str, Any]:
        """Validate API key formats."""
        issues = []

        # Binance API key should be 64 characters
        binance_key = os.getenv('BINANCE_API_KEY', '')
        if len(binance_key) < 64:
            issues.append(f"Binance API key too short ({len(binance_key)} chars, expected 64)")

        # Telegram token format: number:alphanumeric
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        if ':' not in telegram_token:
            issues.append("Telegram token invalid format (expected number:token)")

        if issues:
            return {
                'status': 'WARNING',
                'message': 'API key format issues detected',
                'issues': issues
            }

        return {
            'status': 'PASSED',
            'message': 'All API keys have valid formats'
        }

    async def _check_binance_permissions(self) -> Dict[str, Any]:
        """
        Check Binance API key permissions.

        CRITICAL: Withdrawal MUST be disabled!
        """
        try:
            from src.api_key_manager import get_api_key_manager

            key_manager = get_api_key_manager()
            validation = await key_manager.validate_binance_permissions()

            if not validation.get('valid'):
                if validation.get('has_withdrawal'):
                    return {
                        'status': 'FAILED',
                        'message': '🚨 CRITICAL: Withdrawal permissions ENABLED!',
                        'has_trading': validation.get('has_trading'),
                        'has_withdrawal': True,
                        'permissions': validation.get('permissions', [])
                    }
                else:
                    return {
                        'status': 'WARNING',
                        'message': 'Could not validate permissions',
                        'error': validation.get('error')
                    }

            return {
                'status': 'PASSED',
                'message': '✅ Binance permissions valid (trading enabled, withdrawal disabled)',
                'has_trading': validation.get('has_trading'),
                'has_withdrawal': False,
                'permissions': validation.get('permissions', [])
            }

        except Exception as e:
            logger.error(f"Binance permission check failed: {e}")
            return {
                'status': 'WARNING',
                'message': f'Permission check error: {e}'
            }

    def _check_database_security(self) -> Dict[str, Any]:
        """Check database connection security."""
        db_url = os.getenv('DATABASE_URL', '')

        issues = []

        # Check for localhost in production
        if 'localhost' in db_url.lower() or '127.0.0.1' in db_url:
            issues.append("Database URL points to localhost (may be OK for dev)")

        # Check for weak passwords
        if 'password=admin' in db_url.lower() or 'password=123' in db_url.lower():
            issues.append("🚨 Weak database password detected!")

        # Check SSL mode
        if 'sslmode=require' not in db_url.lower() and 'localhost' not in db_url.lower():
            issues.append("Database connection does not require SSL (production risk)")

        if issues:
            severity = 'FAILED' if any('🚨' in issue for issue in issues) else 'WARNING'
            return {
                'status': severity,
                'message': 'Database security issues',
                'issues': issues
            }

        return {
            'status': 'PASSED',
            'message': 'Database security checks passed'
        }

    def _check_redis_security(self) -> Dict[str, Any]:
        """Check Redis connection security."""
        redis_url = os.getenv('REDIS_URL', '')

        issues = []

        # Check for default Redis port without auth
        if 'redis://localhost:6379' == redis_url or 'redis://127.0.0.1:6379' == redis_url:
            issues.append("Redis without authentication (OK for dev, risky for prod)")

        if issues:
            return {
                'status': 'WARNING',
                'message': 'Redis security warnings',
                'issues': issues
            }

        return {
            'status': 'PASSED',
            'message': 'Redis security checks passed'
        }

    def _update_counts(self, results: Dict[str, Any], status: str):
        """Update pass/fail/warn counts."""
        if status == 'PASSED':
            results['checks_passed'] += 1
        elif status == 'FAILED':
            results['checks_failed'] += 1
        elif status == 'WARNING':
            results['checks_warned'] += 1

    async def _send_validation_summary(self, results: Dict[str, Any]):
        """Send security validation summary via Telegram."""
        try:
            notifier = get_notifier()

            # Build message
            status_emoji = '✅' if results['checks_failed'] == 0 else '🚨'
            message = (
                f"{status_emoji} SECURITY VALIDATION\n\n"
                f"✅ Passed: {results['checks_passed']}\n"
                f"⚠️ Warnings: {results['checks_warned']}\n"
                f"❌ Failed: {results['checks_failed']}\n\n"
            )

            # Add critical issues
            critical_issues = []
            for check_name, check_result in results['details'].items():
                if check_result['status'] == 'FAILED':
                    critical_issues.append(f"❌ {check_name}: {check_result['message']}")

            if critical_issues:
                message += "CRITICAL ISSUES:\n" + '\n'.join(critical_issues)
            else:
                message += "All critical security checks passed!"

            # Rate limit check
            if self.rate_limiter.can_call('telegram'):
                await notifier.send_alert(
                    'success' if results['checks_failed'] == 0 else 'critical',
                    message
                )
                self.rate_limiter.record_call('telegram')

        except Exception as e:
            logger.error(f"Failed to send validation summary: {e}")


class SecurityError(Exception):
    """Raised when critical security check fails."""
    pass


# Singleton instance
_security_validator: Optional[SecurityValidator] = None
_rate_limiter: Optional[RateLimiter] = None


def get_security_validator() -> SecurityValidator:
    """Get or create security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
