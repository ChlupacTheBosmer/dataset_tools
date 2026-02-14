"""High-level orchestration for FiftyOne <-> Label Studio curation roundtrip.

The roundtrip workflow composes three operations:

1. send tagged samples to Label Studio
2. pull submitted annotations back into FiftyOne
3. optionally sync corrected labels back to disk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataset_tools.config import AppConfig
from dataset_tools.tag_workflow import TagOperationRule, TagWorkflowConfig, TagWorkflowEngine


@dataclass(frozen=True)
class RoundtripWorkflowConfig:
    """Configuration for a single roundtrip execution.

    This object captures both control flags (which stages to run) and
    stage-specific parameter dictionaries that are forwarded into underlying
    tag-workflow operations.
    """
    dataset_name: str
    send_tag: str = "fix"
    project_title: str | None = None
    label_field: str = "ground_truth"
    corrections_field: str = "ls_corrections"
    send_to_label_studio: bool = True
    pull_from_label_studio: bool = True
    sync_to_disk: bool = True
    dry_run_sync: bool = False
    clear_project_tasks: bool = False
    upload_strategy: str | None = None
    additional_send_params: dict[str, Any] = field(default_factory=dict)
    additional_pull_params: dict[str, Any] = field(default_factory=dict)
    additional_sync_params: dict[str, Any] = field(default_factory=dict)


class CurationRoundtripWorkflow:
    """Build and execute roundtrip workflows via the tag-workflow engine."""

    def __init__(self, app_config: AppConfig):
        """Create a roundtrip workflow runner bound to resolved app config.

        Args:
            app_config: Resolved application config used by all workflow stages.
        """
        self.app_config = app_config
        self.engine = TagWorkflowEngine(app_config=app_config)

    def run(self, config: RoundtripWorkflowConfig) -> list[dict[str, Any]]:
        """Build stage rules from ``config`` and execute them in order.

        Rule order is deterministic:
        - ``send_to_label_studio`` (optional)
        - ``pull_from_label_studio`` (optional)
        - ``sync_corrections_to_disk`` (optional)

        Args:
            config: Roundtrip run configuration.

        Returns:
            Ordered list of operation result payloads from the workflow engine.
        """
        rules: list[TagOperationRule] = []

        if config.send_to_label_studio:
            send_params = {
                "project_title": config.project_title or self.app_config.label_studio.project_title,
                "label_field": config.label_field,
                "clear_project_tasks": config.clear_project_tasks,
                **config.additional_send_params,
            }
            if config.upload_strategy:
                send_params["upload_strategy"] = config.upload_strategy

            rules.append(
                TagOperationRule(
                    tag=config.send_tag,
                    operation="send_to_label_studio",
                    params=send_params,
                )
            )

        if config.pull_from_label_studio:
            pull_params = {
                "project_title": config.project_title or self.app_config.label_studio.project_title,
                "corrections_field": config.corrections_field,
                **config.additional_pull_params,
            }
            if config.upload_strategy:
                pull_params["upload_strategy"] = config.upload_strategy
            rules.append(
                TagOperationRule(
                    tag=None,
                    operation="pull_from_label_studio",
                    params=pull_params,
                )
            )

        if config.sync_to_disk:
            sync_params = {
                "corrections_field": config.corrections_field,
                "dry_run": config.dry_run_sync,
                **config.additional_sync_params,
            }
            rules.append(
                TagOperationRule(
                    tag=config.send_tag,
                    operation="sync_corrections_to_disk",
                    params=sync_params,
                )
            )

        workflow_config = TagWorkflowConfig(
            dataset_name=config.dataset_name,
            rules=rules,
            fail_fast=True,
        )
        return self.engine.run(workflow_config)
