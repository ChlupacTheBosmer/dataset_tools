"""Implementation module for tag workflow execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TagOperationRule:
    """TagOperationRule used by tag workflow execution.
    """
    operation: str
    tag: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TagWorkflowConfig:
    """Configuration dataclass for tag workflow execution.
    """
    dataset_name: str
    rules: list[TagOperationRule]
    fail_fast: bool = True
