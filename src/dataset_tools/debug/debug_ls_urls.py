"""Implementation module for debug and diagnostics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset_tools.config import load_config


def main():
    """Perform main.

Returns:
    Process exit code.
    """
    cfg = load_config()
    host_root = cfg.mount.host_root
    base_url = cfg.label_studio.url.rstrip("/")

    print(f"Host root: {host_root}")
    print(f"LS URL: {base_url}")

    # Demo path conversion logic
    sample_file = Path(host_root) / "frames_visitors_ndb" / "images" / "CZ" / "sample.jpg"
    rel_path = os.path.relpath(str(sample_file), host_root)
    ls_path = f"{cfg.mount.local_files_prefix}{rel_path}"

    print(f"Sample host path: {sample_file}")
    print(f"Converted LS local-files path: {ls_path}")
    print(f"Absolute URL: {base_url}{ls_path}")


if __name__ == "__main__":
    main()
