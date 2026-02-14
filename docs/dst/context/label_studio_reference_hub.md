# Label Studio Reference Hub

This is the canonical external reference index for Label Studio when working on `dataset_tools`.

Use this file instead of large endpoint dumps or local path references.

## Official Documentation Entry Points

- Product docs: https://labelstud.io/guide/
- API docs home: https://api.labelstud.io/
- API reference root: https://api.labelstud.io/api-reference/
- SDK package: https://pypi.org/project/label-studio-sdk/

## API/SDK Topics Most Relevant To `dataset_tools`

### Authentication and tokens

- API getting started: https://api.labelstud.io/api-reference/introduction/getting-started
- User token endpoints (whoami/token): https://api.labelstud.io/api-reference/api-reference/users/

### Project lifecycle

- Projects endpoints: https://api.labelstud.io/api-reference/api-reference/projects/
- Project creation tutorial: https://api.labelstud.io/tutorials/tutorials/create-a-project

### Task upload and retrieval

- Import tasks tutorial: https://api.labelstud.io/tutorials/tutorials/import-tasks
- Tasks endpoints: https://api.labelstud.io/api-reference/api-reference/tasks/
- Export/snapshot tutorial: https://api.labelstud.io/tutorials/tutorials/export-and-convert-snapshots

### Storage setup

- Import storage endpoints: https://api.labelstud.io/api-reference/api-reference/import-storage/
- Export storage endpoints: https://api.labelstud.io/api-reference/api-reference/export-storage/

## How This Maps To `dst`

- `dst ls ...`: connectivity, project lookup/list/cleanup helpers
- `dst workflow roundtrip`: send selected view tasks, pull submitted annotations, optional disk sync
- `dataset_tools.label_studio.storage`: idempotent storage bootstrap
- `dataset_tools.label_studio.sync`: send/pull bridging with sample mapping

## Agent Notes

- Prefer SDK usage over ad hoc HTTP calls for stable project/task operations.
- Keep upload behavior aligned with tested batched flow (`sdk_batched`).
- Preserve task metadata mapping (`fiftyone_id`) so pull can update the correct samples.
