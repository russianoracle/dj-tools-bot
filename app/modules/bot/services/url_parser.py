"""URL parsing and validation service for batch URL processing."""

import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class ParsedURL:
    """Represents a parsed and validated URL."""
    url: str
    platform: str
    is_valid: bool
    error: Optional[str] = None


class URLParser:
    """Service for extracting and validating URLs from messages."""

    # Platform mapping
    PLATFORM_NAMES = {
        'soundcloud.com': 'SoundCloud',
        'mixcloud.com': 'Mixcloud',
        'youtube.com': 'YouTube',
        'youtu.be': 'YouTube',
        'm.youtube.com': 'YouTube',
    }

    # Allowed domains for SSRF protection
    ALLOWED_DOMAINS = [
        'soundcloud.com',
        'mixcloud.com',
        'youtube.com',
        'youtu.be',
        'm.youtube.com',
    ]

    # URL extraction pattern
    URL_PATTERN = r'https?://[^\s]+'

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract all URLs from text.

        Args:
            text: Message text to parse

        Returns:
            List of extracted URLs
        """
        return re.findall(self.URL_PATTERN, text, re.IGNORECASE)

    def validate_url(self, url: str) -> ParsedURL:
        """
        Validate URL against platform allowlist and SSRF protection.

        Args:
            url: URL to validate

        Returns:
            ParsedURL with validation result
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check if domain matches allowed list
            is_allowed = any(
                domain == d or domain.endswith('.' + d)
                for d in self.ALLOWED_DOMAINS
            )

            if not is_allowed:
                return ParsedURL(
                    url=url,
                    platform='Unknown',
                    is_valid=False,
                    error='Unsupported platform (only SoundCloud/Mixcloud/YouTube allowed)'
                )

            # Determine platform name
            platform = 'Unknown'
            for domain_key, platform_name in self.PLATFORM_NAMES.items():
                if domain == domain_key or domain.endswith('.' + domain_key):
                    platform = platform_name
                    break

            return ParsedURL(
                url=url,
                platform=platform,
                is_valid=True,
                error=None
            )

        except Exception as e:
            return ParsedURL(
                url=url,
                platform='Unknown',
                is_valid=False,
                error=f'Invalid URL format: {str(e)}'
            )

    def parse_batch(self, text: str) -> List[ParsedURL]:
        """
        Extract and validate all URLs from text.

        Args:
            text: Message text to parse

        Returns:
            List of ParsedURL objects with validation results
        """
        urls = self.extract_urls(text)
        return [self.validate_url(url) for url in urls]
