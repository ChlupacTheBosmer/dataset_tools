"""Implementation module for dataset loading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fiftyone as fo  # type: ignore

from dataset_tools.loaders.base import BaseDatasetLoader, LoaderResult
from dataset_tools.loaders.path_resolvers import (
    ImagesLabelsSubdirResolver,
    MirroredRootsPathResolver,
    default_image_filter,
)


@dataclass(frozen=True)
class YoloParserConfig:
    """Configuration dataclass for dataset loading.
    """
    class_id_to_label: dict[int, str] = field(default_factory=dict)
    include_confidence: bool = True


class YoloDatasetLoader(BaseDatasetLoader):
    """Dataset loader that imports source media/annotations into FiftyOne.
    """
    def __init__(
        self,
        resolver: MirroredRootsPathResolver | ImagesLabelsSubdirResolver,
        parser_config: YoloParserConfig | None = None,
        image_filter: Callable[[Path], bool] = default_image_filter,
        sample_metadata_fields: dict[str, Callable[[Path], object]] | None = None,
    ):
        """Initialize `YoloDatasetLoader` with runtime parameters.

Args:
    resolver: Value controlling resolver for this routine.
    parser_config: Value controlling parser config for this routine.
    image_filter: Value controlling image filter for this routine.
    sample_metadata_fields: Value controlling sample metadata fields for this routine.

Returns:
    None.
        """
        self.resolver = resolver
        self.parser_config = parser_config or YoloParserConfig()
        self.image_filter = image_filter
        self.sample_metadata_fields = sample_metadata_fields or {}

    def load(self, dataset_name: str, overwrite: bool = False, persistent: bool = True) -> LoaderResult:
        """Perform load.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    overwrite: Whether existing resources should be replaced.
    persistent: Whether created FiftyOne datasets should be persistent.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        dataset = self._create_or_replace_dataset(dataset_name, overwrite=overwrite, persistent=persistent)
        images_root = self.resolver.images_root

        samples = []
        for image_path in images_root.rglob("*"):
            if not image_path.is_file() or not self.image_filter(image_path):
                continue

            label_path = self.resolver.label_path_for(image_path)

            sample = fo.Sample(filepath=str(image_path))
            for field_name, getter in self.sample_metadata_fields.items():
                sample[field_name] = getter(image_path)

            detections = self._parse_yolo_file(label_path)
            if detections is not None:
                sample["ground_truth"] = detections

            samples.append(sample)

        dataset.add_samples(samples)
        return LoaderResult(dataset_name=dataset.name, sample_count=len(samples))

    def _parse_yolo_file(self, label_path: Path):
        """Parse and validate yolo file input values.

Args:
    label_path: Value controlling label path for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if not label_path.exists():
            return None

        detections = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(float(parts[0]))
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                top_left_x = center_x - width / 2
                top_left_y = center_y - height / 2

                label = self.parser_config.class_id_to_label.get(cls_id, str(cls_id))
                kwargs = {
                    "label": label,
                    "bounding_box": [top_left_x, top_left_y, width, height],
                }

                if self.parser_config.include_confidence and len(parts) > 5:
                    kwargs["confidence"] = float(parts[5])

                detections.append(fo.Detection(**kwargs))

        return fo.Detections(detections=detections)
