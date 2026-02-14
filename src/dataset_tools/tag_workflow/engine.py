"""Implementation module for tag workflow execution.
"""
from __future__ import annotations

from typing import Any

import fiftyone as fo  # type: ignore

from dataset_tools.config import AppConfig
from dataset_tools.tag_workflow.config import TagWorkflowConfig
from dataset_tools.tag_workflow.context import TagWorkflowContext
from dataset_tools.tag_workflow.operations.core import default_operations_registry


class TagWorkflowEngine:
    """TagWorkflowEngine used by tag workflow execution.
    """
    def __init__(self, app_config: AppConfig, operations: dict[str, Any] | None = None):
        """Initialize `TagWorkflowEngine` with runtime parameters.

Args:
    app_config: Resolved `AppConfig` instance.
    operations: Value controlling operations for this routine.

Returns:
    None.
        """
        self.app_config = app_config
        self.operations = operations or default_operations_registry()

    def register_operation(self, name: str, operation):
        """Perform register operation.

Args:
    name: Name identifier for the resource being created or retrieved.
    operation: Value controlling operation for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        self.operations[name] = operation

    def run(self, workflow_config: TagWorkflowConfig) -> list[dict[str, Any]]:
        """Run the operation and return execution results.

Args:
    workflow_config: Value controlling workflow config for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if workflow_config.dataset_name not in fo.list_datasets():
            raise RuntimeError(f"Dataset '{workflow_config.dataset_name}' not found")

        dataset = fo.load_dataset(workflow_config.dataset_name)
        context = TagWorkflowContext(
            dataset=dataset,
            dataset_name=workflow_config.dataset_name,
            app_config=self.app_config,
        )

        results: list[dict[str, Any]] = []

        for rule in workflow_config.rules:
            operation = self.operations.get(rule.operation)
            if operation is None:
                raise RuntimeError(f"Unknown operation '{rule.operation}'")

            view = dataset if rule.tag is None else dataset.match_tags(rule.tag)

            try:
                result = operation.execute(
                    context=context,
                    view=view,
                    params=dict(rule.params),
                    tag=rule.tag,
                )
                results.append(result)
            except Exception:
                if workflow_config.fail_fast:
                    raise
                results.append(
                    {
                        "operation": rule.operation,
                        "tag": rule.tag,
                        "error": True,
                    }
                )

        return results
