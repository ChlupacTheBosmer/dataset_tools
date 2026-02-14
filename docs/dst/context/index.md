# DST Context Docs

This folder contains supporting context for `dataset_tools` (`dst`) that is useful for operators and AI/code agents.

These are not API/module docs; they are workflow, external-reference, and integration notes used to make correct implementation decisions.

## Contents

- `docs/dst/context/fiftyone_reference_hub.md`
  - curated map of official FiftyOne docs relevant to current `dst` features
- `docs/dst/context/label_studio_reference_hub.md`
  - curated Label Studio API/SDK references relevant to the implemented integration
- `docs/dst/context/label_studio_integration_notes.md`
  - path-mapping, local-files, and token/auth notes for robust LS operation
- `docs/dst/context/workflow_hygiene_guide.md`
  - production curation workflow (FiftyOne -> Label Studio -> FiftyOne -> disk sync)
- `docs/dst/context/advanced_analysis_research.md`
  - consolidated research notes on embeddings/brain/anomaly methods and how they map to current architecture
- `docs/dst/context/manual_validation_checklist.md`
  - manual validation checklist for real-world execution safety and behavior checks
- `docs/dst/context/cross_repo_compatibility_matrix.md`
  - shared dependency baseline when using one venv for `dst` + `od_training`
- `docs/dst/context/od_training_handoff_contract.md`
  - required export contract for seamless training handoff

## Scope Rules

- Prefer official web docs over local file paths.
- Keep this folder focused on stable operational knowledge.
- Move experiment-specific notes to cluster-specific docs, not here.
