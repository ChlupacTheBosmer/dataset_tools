"""Implementation module for debug and diagnostics.
"""

import sys
print(f"Python: {sys.executable}")
print("Importing fiftyone...")
try:
    import fiftyone as fo
    print(f"FiftyOne version: {fo.__version__}")
except Exception as e:
    print(f"Failed to import: {e}")
print("Done.")
