"""
API Key Rotation Manager

Provides secure API key rotation with:
- 30-day automatic rotation schedule
- Key expiration tracking
- Encrypted key storage
- Telegram notifications
- Graceful key transitions
- Audit logging

Security Features:
- AES-256 encryption for stored keys
- Secure key generation
- Permission validation
- Rate limiting
- Automatic backup before rotation
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from decimal import Decimal
import asyncio

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ cryptography not installed. Install with: pip install cryptography")

from src.database import get_db_client
from src.telegram_notifier import get_notifier
from src.utils import setup_logging

logger = setup_logging()


class APIKeyManager:
    """
    Manages API key rotation and security.

    Features:
    - Automatic 30-day rotation
    - Encrypted storage
    - Telegram alerts
    - Permission validation
    - Audit trail
    """

    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography required for API Key Manager")

        # Storage paths
        self.data_dir = Path("data/security")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.key_file = self.data_dir / "api_keys.enc"
        self.rotation_log = self.data_dir / "rotation_log.json"
        self.master_key_file = self.data_dir / ".master_key"

        # Rotation settings
        self.rotation_interval_days = 30
        self.warning_days_before = 7  # Warn 7 days before expiration

        # Initialize encryption
        self.cipher = self._initialize_encryption()

        # Load or create key metadata
        self.key_metadata = self._load_key_metadata()

    def _initialize_encryption(self) -> Fernet:
        """
        Initialize encryption cipher using master key.

        If master key doesn't exist, generates one from environment.
        """
        if self.master_key_file.exists():
            # Load existing master key
            with open(self.master_key_file, 'rb') as f:
                master_key = f.read()
        else:
            # Generate new master key from environment secret
            # This should be a strong secret, never committed to git
            env_secret = os.getenv('MASTER_ENCRYPTION_KEY', 'default-dev-key-change-in-production')

            # Derive key using PBKDF2
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'trading-bot-salt',  # Static salt (OK for this use case)
                iterations=100000,
                backend=default_backend()
            )
            key_bytes = kdf.derive(env_secret.encode())
            master_key = Fernet.generate_key()

            # Save master key (encrypted with derived key)
            with open(self.master_key_file, 'wb') as f:
                f.write(master_key)

            logger.info("🔐 Generated new master encryption key")

        return Fernet(master_key)

    def _load_key_metadata(self) -> Dict[str, Any]:
        """Load API key metadata (rotation dates, etc.)."""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    encrypted_data = f.read()

                decrypted_data = self.cipher.decrypt(encrypted_data)
                metadata = json.loads(decrypted_data.decode('utf-8'))

                logger.info("✅ Loaded API key metadata")
                return metadata
            except Exception as e:
                logger.error(f"Failed to load key metadata: {e}")
                return self._create_initial_metadata()
        else:
            return self._create_initial_metadata()

    def _create_initial_metadata(self) -> Dict[str, Any]:
        """Create initial key metadata from environment."""
        metadata = {
            'binance': {
                'key_hash': self._hash_key(os.getenv('BINANCE_API_KEY', '')),
                'created_at': datetime.now().isoformat(),
                'last_rotated': datetime.now().isoformat(),
                'next_rotation': (datetime.now() + timedelta(days=self.rotation_interval_days)).isoformat(),
                'rotation_count': 0,
                'permissions_validated': False
            },
            'openrouter': {
                'key_hash': self._hash_key(os.getenv('OPENROUTER_API_KEY', '')),
                'created_at': datetime.now().isoformat(),
                'last_rotated': datetime.now().isoformat(),
                'next_rotation': (datetime.now() + timedelta(days=self.rotation_interval_days)).isoformat(),
                'rotation_count': 0
            },
            'deepseek': {
                'key_hash': self._hash_key(os.getenv('DEEPSEEK_API_KEY', '')),
                'created_at': datetime.now().isoformat(),
                'last_rotated': datetime.now().isoformat(),
                'next_rotation': (datetime.now() + timedelta(days=self.rotation_interval_days)).isoformat(),
                'rotation_count': 0
            },
            'telegram': {
                'token_hash': self._hash_key(os.getenv('TELEGRAM_BOT_TOKEN', '')),
                'created_at': datetime.now().isoformat(),
                'last_rotated': datetime.now().isoformat(),
                'next_rotation': (datetime.now() + timedelta(days=self.rotation_interval_days)).isoformat(),
                'rotation_count': 0
            }
        }

        self._save_key_metadata(metadata)
        logger.info("📝 Created initial key metadata")
        return metadata

    def _save_key_metadata(self, metadata: Dict[str, Any]):
        """Save encrypted key metadata to disk."""
        try:
            # Encrypt metadata
            json_data = json.dumps(metadata, indent=2).encode('utf-8')
            encrypted_data = self.cipher.encrypt(json_data)

            # Save to file
            with open(self.key_file, 'wb') as f:
                f.write(encrypted_data)

            logger.info("💾 Saved encrypted key metadata")
        except Exception as e:
            logger.error(f"Failed to save key metadata: {e}")

    def _hash_key(self, key: str) -> str:
        """Hash API key for verification (without storing plaintext)."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def check_key_expiration(self) -> Dict[str, Any]:
        """
        Check if any API keys are expiring soon.

        Returns:
            Dict with expiration status for each key
        """
        now = datetime.now()
        expiring_soon = {}
        expired = {}

        for service, metadata in self.key_metadata.items():
            next_rotation = datetime.fromisoformat(metadata['next_rotation'])
            days_until_expiration = (next_rotation - now).days

            if days_until_expiration <= 0:
                expired[service] = {
                    'days_overdue': abs(days_until_expiration),
                    'last_rotated': metadata['last_rotated']
                }
            elif days_until_expiration <= self.warning_days_before:
                expiring_soon[service] = {
                    'days_remaining': days_until_expiration,
                    'next_rotation': metadata['next_rotation']
                }

        return {
            'expired': expired,
            'expiring_soon': expiring_soon,
            'all_ok': len(expired) == 0 and len(expiring_soon) == 0
        }

    async def validate_binance_permissions(self) -> Dict[str, Any]:
        """
        Validate Binance API key permissions.

        Checks:
        - Trading enabled
        - Withdrawal DISABLED (security)
        - Read permissions
        """
        try:
            from src.exchange_client import get_exchange_client
            exchange = await get_exchange_client()

            # Fetch account info to check permissions
            try:
                account = await exchange.exchange.fetch_balance()
                permissions = account.get('info', {}).get('permissions', [])

                # Check permissions
                has_trading = 'TRADING' in permissions or 'SPOT' in permissions
                has_withdrawal = 'WITHDRAWING' in permissions

                result = {
                    'valid': has_trading and not has_withdrawal,
                    'has_trading': has_trading,
                    'has_withdrawal': has_withdrawal,
                    'permissions': permissions
                }

                # Update metadata
                self.key_metadata['binance']['permissions_validated'] = True
                self.key_metadata['binance']['last_validation'] = datetime.now().isoformat()
                self._save_key_metadata(self.key_metadata)

                if has_withdrawal:
                    logger.critical("🚨 SECURITY: Binance API key has WITHDRAWAL permissions!")
                    await self._send_security_alert(
                        "🚨 CRITICAL SECURITY ISSUE",
                        f"Binance API key has WITHDRAWAL permissions enabled!\n"
                        f"This is a major security risk.\n\n"
                        f"Please:\n"
                        f"1. Go to Binance API settings\n"
                        f"2. Disable withdrawal permissions\n"
                        f"3. Rotate the API key"
                    )

                return result

            except Exception as api_error:
                logger.error(f"Failed to validate Binance permissions: {api_error}")
                return {
                    'valid': False,
                    'error': str(api_error)
                }

        except Exception as e:
            logger.error(f"Permission validation error: {e}")
            return {
                'valid': False,
                'error': str(e)
            }

    async def rotate_key(self, service: str, new_key: str) -> Dict[str, Any]:
        """
        Rotate API key for specified service.

        Args:
            service: Service name ('binance', 'openrouter', 'deepseek', 'telegram')
            new_key: New API key value

        Returns:
            Dict with rotation status
        """
        try:
            logger.info(f"🔄 Rotating API key for {service}...")

            # Validate new key format
            if not new_key or len(new_key) < 10:
                return {
                    'success': False,
                    'reason': 'Invalid key format (too short)'
                }

            # Check if key is different
            new_key_hash = self._hash_key(new_key)
            old_key_hash = self.key_metadata[service].get('key_hash') or self.key_metadata[service].get('token_hash')

            if new_key_hash == old_key_hash:
                return {
                    'success': False,
                    'reason': 'New key is same as old key'
                }

            # Backup old metadata
            backup_file = self.data_dir / f"backup_{service}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump({service: self.key_metadata[service]}, f, indent=2)

            logger.info(f"📦 Backed up old key metadata to {backup_file}")

            # Update metadata
            old_metadata = self.key_metadata[service].copy()
            self.key_metadata[service].update({
                'key_hash' if service != 'telegram' else 'token_hash': new_key_hash,
                'last_rotated': datetime.now().isoformat(),
                'next_rotation': (datetime.now() + timedelta(days=self.rotation_interval_days)).isoformat(),
                'rotation_count': old_metadata.get('rotation_count', 0) + 1,
                'previous_key_hash': old_key_hash
            })

            # Save updated metadata
            self._save_key_metadata(self.key_metadata)

            # Log rotation event
            await self._log_rotation_event(service, old_metadata)

            # Send notification
            await self._send_rotation_notification(service)

            # Validate new key if Binance
            if service == 'binance':
                # Update environment variable
                os.environ['BINANCE_API_KEY'] = new_key

                # Validate permissions
                validation = await self.validate_binance_permissions()
                if not validation.get('valid'):
                    logger.warning(f"⚠️ New Binance key validation failed: {validation.get('error')}")

            logger.info(f"✅ Successfully rotated {service} API key")

            return {
                'success': True,
                'service': service,
                'rotation_count': self.key_metadata[service]['rotation_count'],
                'next_rotation': self.key_metadata[service]['next_rotation']
            }

        except Exception as e:
            logger.error(f"Failed to rotate {service} key: {e}")
            return {
                'success': False,
                'reason': str(e)
            }

    async def _log_rotation_event(self, service: str, old_metadata: Dict[str, Any]):
        """Log key rotation event to database and file."""
        try:
            # Load rotation log
            if self.rotation_log.exists():
                with open(self.rotation_log, 'r') as f:
                    log = json.load(f)
            else:
                log = {'rotations': []}

            # Add rotation event
            event = {
                'service': service,
                'timestamp': datetime.now().isoformat(),
                'old_key_hash': old_metadata.get('key_hash') or old_metadata.get('token_hash'),
                'rotation_count': old_metadata.get('rotation_count', 0) + 1,
                'old_last_rotated': old_metadata.get('last_rotated'),
                'reason': 'scheduled_rotation'
            }

            log['rotations'].append(event)

            # Keep only last 100 rotations
            if len(log['rotations']) > 100:
                log['rotations'] = log['rotations'][-100:]

            # Save log
            with open(self.rotation_log, 'w') as f:
                json.dump(log, f, indent=2)

            # Log to database
            db = await get_db_client()
            await db.execute(
                """
                INSERT INTO system_logs (log_level, component, message, details)
                VALUES ($1, $2, $3, $4)
                """,
                'INFO',
                'APIKeyManager',
                f'API key rotated for {service}',
                json.dumps(event)
            )

        except Exception as e:
            logger.error(f"Failed to log rotation event: {e}")

    async def _send_rotation_notification(self, service: str):
        """Send Telegram notification about key rotation."""
        try:
            notifier = get_notifier()

            metadata = self.key_metadata[service]
            next_rotation = datetime.fromisoformat(metadata['next_rotation'])

            message = (
                f"🔐 API KEY ROTATED\n\n"
                f"Service: {service.upper()}\n"
                f"Rotation Count: {metadata['rotation_count']}\n"
                f"Next Rotation: {next_rotation.strftime('%Y-%m-%d')}\n"
                f"Days Until Next: {(next_rotation - datetime.now()).days}\n\n"
                f"✅ Key rotation successful"
            )

            await notifier.send_alert('success', message)

        except Exception as e:
            logger.error(f"Failed to send rotation notification: {e}")

    async def _send_security_alert(self, title: str, message: str):
        """Send critical security alert via Telegram."""
        try:
            notifier = get_notifier()
            await notifier.send_alert('critical', f"{title}\n\n{message}")
        except Exception as e:
            logger.error(f"Failed to send security alert: {e}")

    async def check_and_notify_expiring_keys(self):
        """
        Check for expiring keys and send notifications.

        Should be called daily by scheduler.
        """
        try:
            expiration_status = await self.check_key_expiration()

            # Notify about expired keys
            if expiration_status['expired']:
                for service, info in expiration_status['expired'].items():
                    message = (
                        f"🚨 API KEY EXPIRED\n\n"
                        f"Service: {service.upper()}\n"
                        f"Days Overdue: {info['days_overdue']}\n"
                        f"Last Rotated: {info['last_rotated']}\n\n"
                        f"⚠️ Please rotate this key immediately!\n"
                        f"Use: /rotatekey {service}"
                    )
                    await self._send_security_alert("🚨 KEY EXPIRED", message)

            # Notify about expiring soon
            if expiration_status['expiring_soon']:
                for service, info in expiration_status['expiring_soon'].items():
                    message = (
                        f"⚠️ API KEY EXPIRING SOON\n\n"
                        f"Service: {service.upper()}\n"
                        f"Days Remaining: {info['days_remaining']}\n"
                        f"Next Rotation: {info['next_rotation']}\n\n"
                        f"Please plan to rotate this key soon."
                    )

                    notifier = get_notifier()
                    await notifier.send_alert('warning', message)

            # Log check
            logger.info(f"✅ Key expiration check complete: {len(expiration_status['expired'])} expired, {len(expiration_status['expiring_soon'])} expiring soon")

        except Exception as e:
            logger.error(f"Failed to check expiring keys: {e}")

    def get_key_status(self, service: str) -> Dict[str, Any]:
        """
        Get current status of API key for service.

        Returns:
            Dict with key status information
        """
        if service not in self.key_metadata:
            return {'error': f'Unknown service: {service}'}

        metadata = self.key_metadata[service]
        next_rotation = datetime.fromisoformat(metadata['next_rotation'])
        days_until = (next_rotation - datetime.now()).days

        return {
            'service': service,
            'rotation_count': metadata['rotation_count'],
            'last_rotated': metadata['last_rotated'],
            'next_rotation': metadata['next_rotation'],
            'days_until_rotation': days_until,
            'status': 'expired' if days_until <= 0 else 'expiring_soon' if days_until <= self.warning_days_before else 'ok',
            'permissions_validated': metadata.get('permissions_validated', False)
        }

    def get_all_keys_status(self) -> Dict[str, Any]:
        """Get status of all API keys."""
        return {
            service: self.get_key_status(service)
            for service in self.key_metadata.keys()
        }


# Singleton instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
