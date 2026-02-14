"""Implementation module for Label Studio integration.
"""
from __future__ import annotations

from typing import Any


def _safe_percent(value: float) -> float:
    """Internal helper for safe percent.

Args:
    value: Input value to normalize or validate.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    return max(0.0, min(100.0, value))


def fo_detection_to_ls_result(det: Any, default_label: str = "Insect") -> dict[str, Any]:
    """Perform fo detection to ls result.

Args:
    det: Value controlling det for this routine.
    default_label: Value controlling default label for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    x, y, w, h = det.bounding_box
    label = det.label if getattr(det, "label", None) else default_label

    result = {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": _safe_percent(x * 100),
            "y": _safe_percent(y * 100),
            "width": _safe_percent(w * 100),
            "height": _safe_percent(h * 100),
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }

    confidence = getattr(det, "confidence", None)
    if confidence is not None:
        result["score"] = confidence

    return result


def ls_rectangle_result_to_fo_detection(result: dict[str, Any]):
    """Perform ls rectangle result to fo detection.

Args:
    result: Value controlling result for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    import fiftyone as fo  # type: ignore

    value = result["value"]
    x = float(value["x"]) / 100.0
    y = float(value["y"]) / 100.0
    w = float(value["width"]) / 100.0
    h = float(value["height"]) / 100.0

    label_values = value.get("rectanglelabels") or []
    label = label_values[0] if label_values else "Unknown"

    return fo.Detection(label=label, bounding_box=[x, y, w, h])
