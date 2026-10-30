"""
Shared deployment slug conventions for catalog validation and management.
"""

from __future__ import annotations


def is_group_slug(slug: str) -> bool:
    """
    Identify the reserved group prefix without checking catalog membership.
    """
    return slug.startswith("group_")
