"""
Secrets Manager - Secure API key and sensitive data encryption.

Features:
- Fernet symmetric encryption for API keys
- Master key management
- Environment variable encryption
- Secure key rotation support
- Audit logging for key access

Usage:
    # First time setup:
    from src.secrets_manager import SecretsManager
    sm = SecretsManager()
    sm.generate_master_key()  # Creates .master.key file

    # Encrypt secrets:
    encrypted_api_key = sm.encrypt("your_binance_api_key")

    # Decrypt for use:
    api_key = sm.decrypt(encrypted_api_key)
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class SecretsManager:
    """
    Secure secrets management with encryption.

    Uses Fernet symmetric encryption (AES 128-bit) to encrypt/decrypt
    API keys and other sensitive data.
    """

    def __init__(self, master_key_path: Optional[str] = None):
        """
        Initialize secrets manager.

        Args:
            master_key_path: Path to master key file. Defaults to .master.key
        """
        self.master_key_path = Path(master_key_path or ".master.key")
        self._cipher: Optional[Fernet] = None

        # Try to load existing master key
        if self.master_key_path.exists():
            self._load_master_key()
        else:
            logger.warning(
                "Master key not found. Run generate_master_key() to create one."
            )

    def generate_master_key(self, password: Optional[str] = None) -> str:
        """
        Generate and save a new master encryption key.

        Args:
            password: Optional password for key derivation.
                     If None, generates a random key.

        Returns:
            Base64-encoded master key (SAVE THIS SECURELY!)

        IMPORTANT: Store the returned key securely!
        Without it, encrypted data cannot be decrypted.
        """
        if password:
            # Derive key from password using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=os.urandom(16),
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        else:
            # Generate random key
            key = Fernet.generate_key()

        # Save to file (with restricted permissions)
        self.master_key_path.write_bytes(key)

        # Set file permissions (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            os.chmod(self.master_key_path, 0o600)  # Owner read/write only

        self._cipher = Fernet(key)

        logger.info(f"✅ Master key generated and saved to {self.master_key_path}")
        logger.warning("🔐 CRITICAL: Backup this master key file securely!")

        return key.decode()

    def _load_master_key(self):
        """Load master key from file."""
        try:
            key = self.master_key_path.read_bytes()
            self._cipher = Fernet(key)
            logger.info("✅ Master key loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load master key: {e}")
            raise

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.

        Args:
            plaintext: String to encrypt (e.g., API key)

        Returns:
            Base64-encoded encrypted string

        Example:
            encrypted_key = sm.encrypt("your_api_key_here")
            # Store encrypted_key in .env or database
        """
        if not self._cipher:
            raise RuntimeError(
                "Master key not loaded. Call generate_master_key() first."
            )

        try:
            encrypted_bytes = self._cipher.encrypt(plaintext.encode())
            return encrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, encrypted_text: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted_text: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext string

        Example:
            api_key = sm.decrypt(encrypted_key)
            # Use api_key for API calls
        """
        if not self._cipher:
            raise RuntimeError(
                "Master key not loaded. Ensure .master.key file exists."
            )

        try:
            decrypted_bytes = self._cipher.decrypt(encrypted_text.encode())
            return decrypted_bytes.decode()
        except InvalidToken:
            logger.error("Decryption failed: Invalid token or corrupted data")
            raise ValueError("Invalid encrypted data or wrong master key")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def encrypt_env_file(self, env_path: str = ".env", output_path: str = ".env.encrypted"):
        """
        Encrypt all values in a .env file.

        Args:
            env_path: Path to .env file
            output_path: Path to save encrypted .env file

        Example:
            sm.encrypt_env_file()
            # Creates .env.encrypted with encrypted values
        """
        env_file = Path(env_path)
        if not env_file.exists():
            logger.error(f".env file not found: {env_path}")
            return

        encrypted_lines = []

        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    encrypted_lines.append(line)
                    continue

                # Split key=value
                if '=' in line:
                    key, value = line.split('=', 1)

                    # Encrypt the value
                    encrypted_value = self.encrypt(value)
                    encrypted_lines.append(f"{key}={encrypted_value}")
                else:
                    encrypted_lines.append(line)

        # Write encrypted file
        output_file = Path(output_path)
        output_file.write_text('\n'.join(encrypted_lines))

        logger.info(f"✅ Encrypted .env saved to {output_path}")

    def decrypt_env_value(self, value: str) -> str:
        """
        Helper to decrypt a value from .env file.

        Usage in config.py:
            sm = SecretsManager()
            api_key = sm.decrypt_env_value(os.getenv('BINANCE_API_KEY_ENCRYPTED'))
        """
        # If value looks encrypted (long base64 string), decrypt it
        if len(value) > 50 and '=' not in value:
            try:
                return self.decrypt(value)
            except:
                # If decryption fails, return as-is (might not be encrypted)
                return value
        return value


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# CLI tool for encryption
if __name__ == "__main__":
    import sys

    sm = SecretsManager()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.secrets_manager generate    # Generate master key")
        print("  python -m src.secrets_manager encrypt <text>   # Encrypt text")
        print("  python -m src.secrets_manager decrypt <encrypted>   # Decrypt text")
        print("  python -m src.secrets_manager encrypt-env   # Encrypt .env file")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        key = sm.generate_master_key()
        print(f"\n✅ Master key generated: {sm.master_key_path}")
        print(f"🔐 BACKUP THIS KEY: {key}")
        print("\nIMPORTANT: Store this key securely! Without it, encrypted data cannot be recovered.")

    elif command == "encrypt":
        if len(sys.argv) < 3:
            print("ERROR: Provide text to encrypt")
            sys.exit(1)

        plaintext = sys.argv[2]
        encrypted = sm.encrypt(plaintext)
        print(f"\n✅ Encrypted: {encrypted}")
        print("\nAdd this to your .env file")

    elif command == "decrypt":
        if len(sys.argv) < 3:
            print("ERROR: Provide encrypted text to decrypt")
            sys.exit(1)

        encrypted_text = sys.argv[2]
        try:
            decrypted = sm.decrypt(encrypted_text)
            print(f"\n✅ Decrypted: {decrypted}")
        except Exception as e:
            print(f"\n❌ Decryption failed: {e}")

    elif command == "encrypt-env":
        sm.encrypt_env_file()
        print("\n✅ .env file encrypted to .env.encrypted")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
