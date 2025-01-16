import os
from dotenv import load_dotenv
from typing import List


class Settings:
    """Manages application settings, loading environment variables from .env."""

    def __init__(self, env_file: str = ".env"):
        """Initialize settings and load environment variables."""
        load_dotenv(env_file)

        # Manually load and parse environment variables
        self.alpha_api_keys = self._parse_list(os.getenv("ALPHA_API_KEYS"))
        self.proxy_list = self._parse_list(os.getenv("PROXY_LIST"))
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")

        # Validate important settings
        if not self.alpha_api_keys:
            raise ValueError("ALPHA_API_KEYS is not set or empty in .env file.")
        if not self.proxy_list:
            raise ValueError("PROXY_LIST is not set or empty in .env file.")

    @staticmethod
    def _parse_list(value: str) -> List[str]:
        """Parse a comma-separated string into a list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]


# Instantiate the settings object
settings = Settings()
