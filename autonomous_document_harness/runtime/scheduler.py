"""Scheduler helpers for safe parallelism (see executor.waves)."""

from .executor import doc_affinity, waves

__all__ = ["waves", "doc_affinity"]
