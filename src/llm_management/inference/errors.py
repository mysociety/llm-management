"""
Inference failures shared by reusable runtimes and domain pipelines.
"""

from __future__ import annotations


class ClassifierUnavailable(RuntimeError):
    pass


class ClassifierBusy(RuntimeError):
    pass


class ClassifierOutputError(RuntimeError):
    pass


class ClassifierInputTooLong(ValueError):
    pass
