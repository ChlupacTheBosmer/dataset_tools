# Agent Onboarding

This section is the first-stop launchpad for any AI/code agent entering this repository for the first time.

## Read Order (mandatory)

1. `README.md`
2. `docs/dst/index.md`
3. `docs/dst/agent/onboarding_launchpad.md`
4. `docs/dst/architecture.md`
5. `docs/dst/module-guide.md`
6. `docs/dst/configuration.md`
7. `docs/dst/cli-guide.md`
8. `docs/dst/test-suite.md`

Then use context hubs as needed:

- `docs/dst/context/fiftyone_reference_hub.md`
- `docs/dst/context/label_studio_reference_hub.md`
- `docs/dst/context/label_studio_integration_notes.md`
- `docs/dst/context/manual_validation_checklist.md`

## Core Expectations for Agents

- Keep `dataset_tools` reusable and dataset-agnostic.
- Do not hardcode rodent/insect-specific behavior into package internals.
- Preserve tested Label Studio batched upload behavior.
- Prefer extending `dst` CLI over adding random standalone scripts.
- Never commit secrets (`local_config.json`, API tokens, private paths).

## Where to Start for Common Tasks

- add/modify functionality: `docs/dst/architecture.md` + `docs/dst/module-guide.md`
- operate workflows: `docs/dst/recipes.md`
- debug LS sync issues: `docs/dst/context/label_studio_integration_notes.md`
- validate changes: `docs/dst/test-suite.md`
