# Label Studio Integration Notes

Operational notes for stable FiftyOne <-> Label Studio operation in `dataset_tools`.

## 1. Token/Auth Model

- Use a valid API access token configured via:
  - `LABEL_STUDIO_API_KEY`, or
  - local config JSON loaded by `load_config()`.
- `dataset_tools.label_studio.client` resolves token variants and validates connectivity.

## 2. Local Files Serving and Path Mapping

`dataset_tools` expects Label Studio to serve local files and relies on host-to-pod path mapping:

- `mount.host_root`: host prefix present in FiftyOne sample filepaths
- `mount.ls_document_root`: corresponding root visible inside Label Studio runtime
- `mount.local_files_prefix`: URL prefix used for local-files links

The integration rewrites each sample filepath into a Label Studio local-files URL. If this mapping is wrong, tasks can be created but image URLs will be broken.

## 3. Storage Behavior

- Source/target storage setup is handled by `dataset_tools.label_studio.storage`.
- Storage initialization is designed to be idempotent.
- Keep source/target paths configurable per deployment via config overrides.

## 4. Upload Strategy

Supported strategies:

- `sdk_batched` (default, recommended)
- `annotate_batched` (legacy compatibility path)

For production workloads, prefer `sdk_batched` because it uses explicit batched SDK uploads and preserves stable task metadata needed for pull.

## 5. Pull Semantics

Pull should only map submitted/labeled tasks back to FiftyOne samples. Correct mapping depends on task metadata containing sample identity (`fiftyone_id`).

Recommended operational pattern:

- use one project per annotation batch/run, or
- clear tasks before reusing project IDs.

This avoids accidental mixing of historical and current runs in pull operations.

## 6. Safety Checks

Before sending large batches:

- run preflight validation
- verify LS connectivity with `dst ls test`
- verify mount mapping with a small sample set first

Before writing corrections to disk:

- run `dst sync disk --dry-run`
- then run write mode once reviewed
