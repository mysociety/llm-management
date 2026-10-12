"""
Local recognizer variants that avoid network access during sanitization.
"""

from __future__ import annotations

from presidio_analyzer.predefined_recognizers import EmailRecognizer
from tldextract import TLDExtract


class OfflineEmailRecognizer(EmailRecognizer):
    """
    Validate email domains against tldextract's bundled public suffix snapshot.
    """

    def __init__(self) -> None:
        """
        Retain Presidio's email patterns while disabling suffix downloads and caching.
        """
        super().__init__()
        self.suffixes = TLDExtract(suffix_list_urls=(), cache_dir=None)

    def validate_result(self, pattern_text: str) -> bool:
        """
        Check the matched domain using only the installed suffix snapshot.
        """
        return self.suffixes(pattern_text).fqdn != ""
