"""Implementation module for debug and diagnostics.
"""
import sys
from pathlib import Path

print("Start debug_imports.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

print("Importing fiftyone.brain...")
try:
    import fiftyone.brain
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Importing dataset_tools.config...")
try:
    import dataset_tools.config
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Importing dataset_tools.loader...")
try:
    import dataset_tools.loader
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Importing dataset_tools.label_studio...")
try:
    import dataset_tools.label_studio
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Done.")
