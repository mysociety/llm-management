"""
Failures from deployment orchestration without an HTTP dependency.
"""

from __future__ import annotations


class LLMManagementError(Exception):
    """
    Base exception for llm-management errors.
    """


class DeploymentNotFoundError(LLMManagementError):
    def __init__(self, slug: str) -> None:
        super().__init__(f"Deployment {slug} not found.")


class DeploymentServiceError(RuntimeError):
    """A lifecycle operation failed with a public, safe explanation.

    The status code preserves the existing API classification of these failures;
    callers outside HTTP can inspect the same reason without importing FastAPI.
    """

    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
