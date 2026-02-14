"""Implementation module for model provider registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KNOWN_PROVIDER_ALIASES = {
    "hf": "hf",
    "huggingface": "hf",
    "transformers": "hf",
    "foz": "foz",
    "fiftyone": "foz",
    "fiftyone_zoo": "foz",
    "zoo": "foz",
    "anomalib": "anomalib",
}


@dataclass(frozen=True)
class ModelRef:
    """ModelRef used by model provider registry.
    """
    provider: str
    model_id: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedModel:
    """LoadedModel used by model provider registry.
    """
    ref: ModelRef
    model: Any
    capabilities: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: str | None) -> bool:
        """Perform supports.

Args:
    capability: Value controlling capability for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if capability is None:
            return True
        return capability in set(self.capabilities)


def normalize_provider(provider: str) -> str:
    """Perform normalize provider.

Args:
    provider: Value controlling provider for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    key = provider.strip().lower()
    if key in KNOWN_PROVIDER_ALIASES:
        return KNOWN_PROVIDER_ALIASES[key]
    raise ValueError(
        f"Unknown model provider '{provider}'. "
        f"Supported providers: {sorted(set(KNOWN_PROVIDER_ALIASES.values()))}"
    )


def parse_model_ref(raw: str, default_provider: str = "hf") -> ModelRef:
    """Parse and normalize model ref.

Args:
    raw: Raw text value from input/CLI that will be parsed.
    default_provider: Value controlling default provider for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    value = str(raw).strip()
    if not value:
        raise ValueError("Model reference cannot be empty")

    if ":" in value:
        provider_raw, model_id = value.split(":", 1)
        provider = normalize_provider(provider_raw)
        model_id = model_id.strip()
    else:
        provider = normalize_provider(default_provider)
        model_id = value

    if not model_id:
        raise ValueError(f"Invalid model reference '{raw}', model id is empty")

    return ModelRef(provider=provider, model_id=model_id)
