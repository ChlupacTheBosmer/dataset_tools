"""Implementation module for debug and diagnostics.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
from dataset_tools.config import load_config
from dataset_tools.label_studio.client import ensure_label_studio_client

def debug_tasks():
    """Perform debug tasks.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    cfg = load_config()
    ls = ensure_label_studio_client(cfg)
    title = "RoundTrip_Test"
    projects = ls.list_projects()
    p = next((p for p in projects if p.title == title), None)
    if not p:
        print("Project not found")
        return
        
    print(f"Project ID: {p.id}")
    tasks = p.get_tasks()
    print(f"Got {len(tasks)} tasks.")
    
    if tasks:
        print("Sample Task Structure:")
        print(json.dumps(tasks[0], indent=2))
    else:
        print("No tasks returned via SDK.")
        
        # Try raw API
        print("Trying raw API...")
        res = ls.make_request("GET", f"/api/projects/{p.id}/tasks")
        print(f"Raw Status: {res.status_code}")
        # print(res.text[:500])

if __name__ == "__main__":
    debug_tasks()
