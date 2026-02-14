"""Implementation module for model provider registry.
"""
from __future__ import annotations

from typing import Any

from dataset_tools.models.base import ModelProvider
from dataset_tools.models.providers.anomalib import AnomalibProvider
from dataset_tools.models.providers.fiftyone_zoo import FiftyOneZooProvider
from dataset_tools.models.providers.huggingface import HuggingFaceProvider
from dataset_tools.models.spec import LoadedModel, ModelRef, normalize_provider, parse_model_ref


def _provider_instances() -> dict[str, ModelProvider]:
    """Internal helper for provider instances.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    return {
        "hf": HuggingFaceProvider(),
        "foz": FiftyOneZooProvider(),
        "anomalib": AnomalibProvider(),
    }


def list_providers() -> list[str]:
    """List available providers.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    return sorted(_provider_instances().keys())


def get_provider(name: str) -> ModelProvider:
    """Perform get provider.

Args:
    name: Name identifier for the resource being created or retrieved.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    normalized = normalize_provider(name)
    providers = _provider_instances()
    return providers[normalized]


def resolve_model_ref(raw: str, default_provider: str = "hf") -> ModelRef:
    """Resolve model ref from provided inputs.

Args:
    raw: Raw text value from input/CLI that will be parsed.
    default_provider: Value controlling default provider for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    return parse_model_ref(raw, default_provider=default_provider)


def load_model(
    raw_model_ref: str,
    *,
    default_provider: str = "hf",
    task: str | None = None,
    capability: str | None = None,
    **kwargs: Any,
) -> LoadedModel:
    """Load model required by this module.

Args:
    raw_model_ref: Value controlling raw model ref for this routine.
    default_provider: Value controlling default provider for this routine.
    task: Value controlling task for this routine.
    capability: Value controlling capability for this routine.
    **kwargs: Additional keyword arguments forwarded to downstream APIs.

Returns:
    Loaded object/data required by downstream workflow steps.
    """
    model_ref = resolve_model_ref(raw_model_ref, default_provider=default_provider)
    provider = get_provider(model_ref.provider)
    loaded = provider.load(model_ref, task=task, **kwargs)

    if capability and not loaded.supports(capability):
        raise RuntimeError(
            f"Loaded model '{raw_model_ref}' from provider '{model_ref.provider}' "
            f"does not support required capability '{capability}'. "
            f"Capabilities: {list(loaded.capabilities)}"
        )

    return loaded


def provider_model_list(
    provider_name: str,
    *,
    contains: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Perform provider model list.

Args:
    provider_name: Value controlling provider name for this routine.
    contains: Value controlling contains for this routine.
    limit: Value controlling limit for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    provider = get_provider(provider_name)
    return provider.list_models(contains=contains, limit=limit)
