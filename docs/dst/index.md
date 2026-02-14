# Dataset Tools (`dst`) Documentation

`dataset_tools` is the production data-curation toolkit in this repo. It provides one operational interface (`dst`) for dataset loading, curation metrics, FiftyOne Brain analyses, Label Studio roundtrip workflows, disk sync, and anomaly tooling.

This docs set is organized so you can move from operations -> architecture -> API details -> external context.

## Start Here

- `docs/dst/cli-guide.md`: production CLI usage patterns and safety notes
- `docs/dst/recipes.md`: command recipes for common workflows
- `docs/dst/configuration.md`: config model and secret handling
- `docs/dst/test-suite.md`: test inventory and execution guidance

## Architecture and Internals

- `docs/dst/architecture.md`: subsystem boundaries and runtime data flow
- `docs/dst/module-guide.md`: module responsibilities and composition
- `docs/dst/resources.md`: high-level external references mapped to implemented features

## Context and Reference Hubs

- `docs/dst/context/index.md`: context docs index
- `docs/dst/context/fiftyone_reference_hub.md`: FiftyOne reference map
- `docs/dst/context/label_studio_reference_hub.md`: Label Studio API/SDK reference map
- `docs/dst/context/advanced_analysis_research.md`: consolidated research notes
- `docs/dst/context/cross_repo_compatibility_matrix.md`: shared dependency matrix with `od_training`
- `docs/dst/context/od_training_handoff_contract.md`: export handoff contract for training

## Agent Onboarding

- `docs/dst/agent/index.md`: agent entrypoint and mandatory read order
- `docs/dst/agent/onboarding_launchpad.md`: first-session protocol, invariants, and validation process

## Generated Exhaustive References

- `docs/dst/cli-reference.md`: complete command/subcommand/argument map generated from `dataset_tools.dst.build_parser()`
- `docs/dst/api-reference.md`: complete module/class/function/method map generated from source

## Regeneration

Reference docs are generated from code and parser state:

```bash
python scripts/generate_dst_docs.py
```

## Quick Operator Smoke Path

```bash
./dst config show
./dst ls test
./dst --help
```

## Safety and Production Notes

- Keep Label Studio credentials out of git: use a local config file and/or environment variables.
- Run destructive paths with preview first:
  - `./dst sync disk --dry-run`
  - `./dst ls project cleanup --dry-run`
- `anomalib` torch artifact loading can deserialize pickled objects. Only enable trusted loading for artifacts from trusted sources.
