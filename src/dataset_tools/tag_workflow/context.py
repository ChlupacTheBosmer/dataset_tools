"""Implementation module for tag workflow execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataset_tools.config import AppConfig


@dataclass
class TagWorkflowContext:
    """TagWorkflowContext used by tag workflow execution.
    """
    dataset: Any
    dataset_name: str
    app_config: AppConfig
    caches: dict[str, Any] = field(default_factory=dict)
