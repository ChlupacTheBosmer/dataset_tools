"""Implementation module for Label Studio integration.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path


def build_tasks(root_dir: Path, ls_root: str):
    """Build tasks for downstream steps.

Args:
    root_dir: Root directory used by resolver or loader logic.
    ls_root: Value controlling ls root for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    tasks = []
    img_dir = root_dir / "images"

    for current_root, _, files in os.walk(img_dir):
        for file_name in files:
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = Path(current_root) / file_name
            rel_path = img_path.relative_to(img_dir)
            label_path = root_dir / "labels" / rel_path.with_suffix(".txt")

            task = {
                "data": {"image": f"{ls_root}/{rel_path.as_posix()}"},
                "predictions": [{"model_version": "one_time_import", "result": []}],
            }

            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        cls, x, y, w, h = map(float, parts[:5])
                        task["predictions"][0]["result"].append(
                            {
                                "id": str(uuid.uuid4())[:8],
                                "type": "rectanglelabels",
                                "from_name": "label",
                                "to_name": "image",
                                "original_width": 100,
                                "original_height": 100,
                                "image_rotation": 0,
                                "value": {
                                    "rotation": 0,
                                    "x": (x - w / 2) * 100,
                                    "y": (y - h / 2) * 100,
                                    "width": w * 100,
                                    "height": h * 100,
                                    "rectanglelabels": [str(int(cls))],
                                },
                            }
                        )

            tasks.append(task)

    return tasks
