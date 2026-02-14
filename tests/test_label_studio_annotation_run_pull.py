from __future__ import annotations

import unittest

from dataset_tools.label_studio.sync import pull_labeled_tasks_from_annotation_run


class _FakeSample(dict):
    def __init__(self, sample_id: str):
        super().__init__()
        self.id = sample_id
        self.saved = False

    def save(self):
        self.saved = True


class _FakeResults:
    def __init__(self, project_id: int, uploaded_tasks: dict[int, str]):
        self.project_id = project_id
        self.uploaded_tasks = uploaded_tasks


class _FakeDataset:
    def __init__(self, samples: dict[str, _FakeSample], results_by_key: dict[str, _FakeResults]):
        self._samples = samples
        self._results_by_key = results_by_key
        self.name = "fake_dataset"

    def load_annotation_results(self, annotation_key: str):
        if annotation_key not in self._results_by_key:
            raise KeyError(annotation_key)
        return self._results_by_key[annotation_key]

    def __getitem__(self, sample_id: str):
        if sample_id not in self._samples:
            raise KeyError(sample_id)
        return self._samples[sample_id]


class _FakeProject:
    def __init__(self, tasks: list[dict]):
        self._tasks = tasks

    def get_tasks(self, selected_ids=None):
        if selected_ids is None:
            return list(self._tasks)
        selected = set(selected_ids)
        return [t for t in self._tasks if t.get("id") in selected]


class _FakeLSClient:
    def __init__(self, projects: dict[int, _FakeProject]):
        self._projects = projects

    def get_project(self, project_id: int):
        return self._projects[project_id]


class AnnotationRunPullTests(unittest.TestCase):
    def test_pull_uses_annotation_run_task_map(self):
        s1 = _FakeSample("s1")
        s2 = _FakeSample("s2")

        results = _FakeResults(
            project_id=42,
            uploaded_tasks={101: "s1", 102: "s2"},
        )

        tasks = [
            {
                "id": 101,
                "annotations": [
                    {
                        "updated_at": "2026-02-12T10:00:00Z",
                        "result": [
                            {
                                "type": "rectanglelabels",
                                "value": {
                                    "rectanglelabels": ["rodent"],
                                    "x": 10,
                                    "y": 20,
                                    "width": 30,
                                    "height": 40,
                                },
                            }
                        ],
                    }
                ],
            },
            {
                "id": 102,
                "annotations": [],
            },
        ]

        dataset = _FakeDataset(
            samples={"s1": s1, "s2": s2},
            results_by_key={"run_a": results},
        )
        ls = _FakeLSClient({42: _FakeProject(tasks)})

        updated = pull_labeled_tasks_from_annotation_run(
            dataset=dataset,
            ls_client=ls,
            annotation_key="run_a",
            corrections_field="ls_corrections",
        )

        self.assertEqual(updated, 1)
        self.assertIn("ls_corrections", s1)
        self.assertTrue(s1.saved)
        self.assertNotIn("ls_corrections", s2)

    def test_missing_annotation_run_raises(self):
        dataset = _FakeDataset(samples={}, results_by_key={})
        ls = _FakeLSClient({})

        with self.assertRaises(RuntimeError):
            pull_labeled_tasks_from_annotation_run(
                dataset=dataset,
                ls_client=ls,
                annotation_key="missing",
            )


if __name__ == "__main__":
    unittest.main()
