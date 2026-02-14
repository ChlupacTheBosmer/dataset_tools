"""Base operation contract for tag-workflow actions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dataset_tools.tag_workflow.context import TagWorkflowContext


class TagOperation(ABC):
    """Abstract interface implemented by all tag-workflow operations."""
    name: str

    @abstractmethod
    def execute(
        self,
        context: TagWorkflowContext,
        view: Any,
        params: dict[str, Any],
        tag: str | None,
    ) -> dict[str, Any]:
        """Execute operation logic for the selected tag/view scope.

        Args:
            context: Shared workflow context (dataset, config, caches).
            view: FiftyOne view selected by the current rule/tag.
            params: Rule-specific parameters.
            tag: Tag value associated with the rule (or ``None``).

        Returns:
            JSON-serializable result payload for workflow reporting.
        """
        pass
