#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _ann_to_str(node):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _func_sig(fn):
    args = []
    posonly = fn.args.posonlyargs
    normal = fn.args.args
    defaults = [None] * (len(normal) - len(fn.args.defaults)) + fn.args.defaults

    for a, d in zip(posonly + normal, [None] * len(posonly) + defaults):
        s = a.arg
        ann = _ann_to_str(a.annotation)
        if ann:
            s += f": {ann}"
        if d is not None:
            s += f" = {ast.unparse(d)}"
        args.append(s)

    if fn.args.vararg:
        a = fn.args.vararg
        s = "*" + a.arg
        ann = _ann_to_str(a.annotation)
        if ann:
            s += f": {ann}"
        args.append(s)
    elif fn.args.kwonlyargs:
        args.append("*")

    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        s = a.arg
        ann = _ann_to_str(a.annotation)
        if ann:
            s += f": {ann}"
        if d is not None:
            s += f" = {ast.unparse(d)}"
        args.append(s)

    if fn.args.kwarg:
        a = fn.args.kwarg
        s = "**" + a.arg
        ann = _ann_to_str(a.annotation)
        if ann:
            s += f": {ann}"
        args.append(s)

    ret = _ann_to_str(fn.returns)
    sig = f"({', '.join(args)})"
    if ret:
        sig += f" -> {ret}"
    return sig


def _build_api_inventory(package_root: Path) -> list[dict]:
    inventory = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.as_posix()
        module = rel[:-3].replace("/", ".")
        if rel.endswith("__init__.py"):
            module = rel[:-12].replace("/", ".")

        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_doc = ast.get_docstring(tree)
        entry = {
            "module": module,
            "file": rel,
            "doc": module_doc,
            "classes": [],
            "functions": [],
        }
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entry["functions"].append(
                    {
                        "name": node.name,
                        "signature": _func_sig(node),
                        "lineno": node.lineno,
                        "doc": ast.get_docstring(node),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                cls = {
                    "name": node.name,
                    "lineno": node.lineno,
                    "doc": ast.get_docstring(node),
                    "methods": [],
                }
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cls["methods"].append(
                            {
                                "name": item.name,
                                "signature": _func_sig(item),
                                "lineno": item.lineno,
                                "doc": ast.get_docstring(item),
                            }
                        )
                entry["classes"].append(cls)
        inventory.append(entry)
    return inventory


def _doc_summary(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.strip().replace("\r\n", "\n")
    first_block = normalized.split("\n\n", 1)[0].strip()
    return " ".join(line.strip() for line in first_block.splitlines())


def _build_cli_inventory():
    from dataset_tools.dst import build_parser

    parser = build_parser()

    def action_info(action):
        return {
            "option_strings": list(action.option_strings),
            "dest": action.dest,
            "required": bool(getattr(action, "required", False)),
            "default": None if action.default is argparse.SUPPRESS else action.default,
            "help": action.help,
            "choices": (
                list(action.choices)
                if getattr(action, "choices", None) is not None
                and not isinstance(action, argparse._SubParsersAction)
                else None
            ),
            "metavar": action.metavar,
            "nargs": action.nargs,
        }

    def walk(p, path):
        handler = getattr(p, "_defaults", {}).get("func")
        node = {
            "path": path,
            "prog": p.prog,
            "description": p.description,
            "handler": getattr(handler, "__name__", None) if handler else None,
            "handler_doc": _doc_summary(getattr(handler, "__doc__", None)) if handler else None,
            "positionals": [],
            "options": [],
            "subcommands": {},
        }
        sub = None
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                sub = action
                continue
            if action.dest in ("help",):
                continue
            info = action_info(action)
            if action.option_strings:
                node["options"].append(info)
            else:
                node["positionals"].append(info)
        if sub:
            for name, subparser in sub.choices.items():
                node["subcommands"][name] = walk(subparser, path + [name])
        return node

    return walk(parser, [])


def _module_role(module: str) -> str:
    roles = {
        "dataset_tools": "Top-level package exports for application config and disk sync entrypoints.",
        "dataset_tools.config": "Configuration dataclasses and layered config loading (defaults -> local JSON -> env -> runtime overrides).",
        "dataset_tools.dst": "Unified CLI entrypoint and command handlers for all dataset_tools capabilities.",
        "dataset_tools.loader": "Legacy convenience wrapper functions around the newer loader abstractions.",
        "dataset_tools.label_studio_json": "Utility for converting YOLO-style labels into Label Studio import task JSON.",
        "dataset_tools.sync_from_fo_to_disk": "Writes corrected FiftyOne detection fields back to YOLO label files on disk with backups.",
        "dataset_tools.workflows.roundtrip": "High-level orchestration of send -> pull -> disk-sync labeling loops.",
        "dataset_tools.anomaly.base": "Dataclass for embedding-distance anomaly references.",
        "dataset_tools.anomaly.pipeline": "Embedding-distance anomaly fit/score pipeline and adapter to anomalib artifact scoring.",
        "dataset_tools.anomaly.anomalib": "Tutorial-aligned anomalib workflow: prepare folder data, train/export artifacts, and score datasets.",
        "dataset_tools.brain.base": "Base abstraction for FiftyOne Brain operations.",
        "dataset_tools.brain.visualization": "Brain visualization run wrapper (UMAP/t-SNE/PCA through FiftyOne APIs).",
        "dataset_tools.brain.similarity": "Brain similarity-index run wrapper.",
        "dataset_tools.brain.duplicates": "Exact and near-duplicate detection wrappers.",
        "dataset_tools.brain.leaky_splits": "Leaky split detection wrapper.",
        "dataset_tools.loaders.base": "Base loader interfaces and shared loader result contract.",
        "dataset_tools.loaders.path_resolvers": "Path resolver primitives that map image paths to annotation paths.",
        "dataset_tools.loaders.yolo": "Recursive YOLO dataset loader (with optional confidence support).",
        "dataset_tools.loaders.coco": "COCO detection dataset loader.",
        "dataset_tools.metrics.base": "Base metric computation abstraction.",
        "dataset_tools.metrics.field_metric": "Shared scaffold for metrics that write sample fields with required-field validation.",
        "dataset_tools.metrics.embeddings": "Embedding computation + optional visualization/clustering enrichment.",
        "dataset_tools.metrics.uniqueness": "Uniqueness score computation wrapper.",
        "dataset_tools.metrics.mistakenness": "Mistakenness/missing/spurious field computation wrapper.",
        "dataset_tools.metrics.hardness": "Hardness computation wrapper for classification-style fields.",
        "dataset_tools.metrics.representativeness": "Representativeness score computation wrapper.",
        "dataset_tools.models.spec": "Provider/model reference dataclasses and parsing helpers.",
        "dataset_tools.models.base": "Provider interface base class.",
        "dataset_tools.models.registry": "Provider registry, provider resolution, and capability validation.",
        "dataset_tools.models.providers.huggingface": "HuggingFace embedding model adapter provider.",
        "dataset_tools.models.providers.fiftyone_zoo": "FiftyOne model zoo provider adapter.",
        "dataset_tools.models.providers.anomalib": "Anomalib model provider adapter.",
        "dataset_tools.label_studio.client": "Label Studio SDK client import/connect/token handling.",
        "dataset_tools.label_studio.storage": "Project and storage bootstrap (source/target local storage).",
        "dataset_tools.label_studio.uploader": "Batched upload monkeypatch for FiftyOne Label Studio integration.",
        "dataset_tools.label_studio.sync": "Send/pull synchronization utilities between FiftyOne and Label Studio.",
        "dataset_tools.label_studio.translator": "BBox format translation between FiftyOne and Label Studio schemas.",
        "dataset_tools.tag_workflow.config": "Rule-based workflow dataclasses.",
        "dataset_tools.tag_workflow.context": "Runtime workflow context object.",
        "dataset_tools.tag_workflow.engine": "Rule execution engine over tag-selected views.",
        "dataset_tools.tag_workflow.operations.base": "Operation interface base class.",
        "dataset_tools.tag_workflow.operations.core": "Core mutation and Label Studio roundtrip operations registry.",
        "dataset_tools.tag_workflow.operations.analysis": "Analysis operations exposed to the tag workflow engine.",
    }
    if module in roles:
        return roles[module]
    if ".debug." in module or module.startswith("dataset_tools.debug"):
        return "Debug/support script; not intended as stable production API."
    return "Internal module in dataset_tools. See source and call graph for behavior."


def _callable_role(module: str, name: str) -> str:
    if name.startswith("_"):
        return "Internal helper"
    if module == "dataset_tools.dst" and name.startswith("cmd_"):
        return "CLI command handler"
    if name in {"build_parser", "main"}:
        return "CLI entrypoint"
    if name.startswith("ensure_") or name.startswith("connect_"):
        return "Integration/bootstrap API"
    if name.startswith("parse_") or name.startswith("normalize_") or name.startswith("resolve_"):
        return "Parsing/resolution utility"
    if name.startswith("compute_"):
        return "Analysis computation API"
    return "Public callable"


def _md_cell(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("|", r"\|").replace("\n", "<br>")


def _render_api_markdown(api_inventory: list[dict]) -> str:
    lines = []
    lines.append("# Dataset Tools API Reference (Complete)")
    lines.append("")
    lines.append(
        "This reference is generated from the current `dataset_tools` source tree and is intended as an exhaustive API map."
    )
    lines.append("")
    lines.append("- Scope: every Python module, top-level function, class, and class method in `dataset_tools/`")
    lines.append("- Stability: names marked \"Internal helper\" are implementation details and can change without notice")
    lines.append("- Narrative: descriptions are sourced from in-code docstrings for maintainable docs-as-code")
    lines.append("")
    lines.append("## Package Inventory")
    lines.append("")
    lines.append("| Module | Role | Module Summary |")
    lines.append("|---|---|---|")
    for entry in api_inventory:
        lines.append(
            f"| `{entry['module']}` | {_md_cell(_module_role(entry['module']))} | {_md_cell(_doc_summary(entry.get('doc')))} |"
        )

    for entry in api_inventory:
        lines.append("")
        lines.append(f"## `{entry['module']}`")
        lines.append("")
        lines.append(f"- File: `{entry['file']}`")
        lines.append(f"- Role: {_module_role(entry['module'])}")
        module_summary = _doc_summary(entry.get("doc"))
        if module_summary:
            lines.append(f"- Module Summary: {module_summary}")

        if not entry["classes"] and not entry["functions"]:
            lines.append("- No top-level classes or functions (export-only or script-only module).")
            continue

        if entry["classes"]:
            lines.append("")
            lines.append("### Classes")
            lines.append("")
            lines.append("| Class | Line | Role | Summary |")
            lines.append("|---|---:|---|---|")
            for cls in entry["classes"]:
                lines.append(
                    f"| `{cls['name']}` | {cls['lineno']} | {_md_cell(_callable_role(entry['module'], cls['name']))} | {_md_cell(_doc_summary(cls.get('doc')))} |"
                )

            for cls in entry["classes"]:
                lines.append("")
                lines.append(f"#### `{cls['name']}` methods")
                class_summary = _doc_summary(cls.get("doc"))
                if class_summary:
                    lines.append(f"- Class Summary: {class_summary}")
                if not cls["methods"]:
                    lines.append("- No methods declared in this class body.")
                    continue
                lines.append("")
                lines.append("| Method | Signature | Line | Role | Summary |")
                lines.append("|---|---|---:|---|---|")
                for method in cls["methods"]:
                    lines.append(
                        f"| `{method['name']}` | `{method['signature']}` | {method['lineno']} | {_md_cell(_callable_role(entry['module'], method['name']))} | {_md_cell(_doc_summary(method.get('doc')))} |"
                    )

        if entry["functions"]:
            lines.append("")
            lines.append("### Functions")
            lines.append("")
            lines.append("| Function | Signature | Line | Role | Summary |")
            lines.append("|---|---|---:|---|---|")
            for fn in entry["functions"]:
                lines.append(
                    f"| `{fn['name']}` | `{fn['signature']}` | {fn['lineno']} | {_md_cell(_callable_role(entry['module'], fn['name']))} | {_md_cell(_doc_summary(fn.get('doc')))} |"
                )

    return "\n".join(lines) + "\n"


def _render_cli_markdown(cli_inventory: dict) -> str:
    def command_examples(path):
        if not path:
            return ["`./dst --help`"]
        return [f"`./dst {' '.join(path)} --help`"]

    def render_command(node, depth=2):
        path = node["path"]
        title = " ".join(path) if path else "dst"
        lines = []
        lines.append("#" * depth + f" `{title}`")
        lines.append("")
        if node.get("description"):
            lines.append(node["description"])
            lines.append("")
        lines.append(f"- Parser prog: `{node['prog']}`")
        for ex in command_examples(path):
            lines.append(f"- Help: {ex}")
        if node.get("handler"):
            lines.append(f"- Handler: `{node['handler']}`")
            if node.get("handler_doc"):
                lines.append(f"- Handler Summary: {_md_cell(node['handler_doc'])}")
        lines.append("")

        if node["positionals"]:
            lines.append("**Positional Arguments**")
            lines.append("")
            lines.append("| Name | Required | Choices | Default | Help |")
            lines.append("|---|---|---|---|---|")
            for arg in node["positionals"]:
                if arg["dest"] == "command":
                    continue
                choices = ", ".join(map(str, arg["choices"])) if arg["choices"] else ""
                default = "" if arg["default"] is None else str(arg["default"])
                lines.append(
                    f"| `{arg['dest']}` | {'yes' if arg['required'] else 'no'} | {choices} | {default} | {arg['help'] or ''} |"
                )
            lines.append("")

        if node["options"]:
            lines.append("**Options**")
            lines.append("")
            lines.append("| Flags | Dest | Required | Choices | Default | Help |")
            lines.append("|---|---|---|---|---|---|")
            for arg in node["options"]:
                flags = ", ".join(arg["option_strings"])
                choices = ", ".join(map(str, arg["choices"])) if arg["choices"] else ""
                default = "" if arg["default"] is None else str(arg["default"])
                lines.append(
                    f"| `{flags}` | `{arg['dest']}` | {'yes' if arg['required'] else 'no'} | {choices} | {default} | {arg['help'] or ''} |"
                )
            lines.append("")

        if node["subcommands"]:
            lines.append("**Subcommands**")
            lines.append("")
            for name in sorted(node["subcommands"]):
                lines.append(f"- `{name}`")
            lines.append("")
            for name in sorted(node["subcommands"]):
                lines.extend(render_command(node["subcommands"][name], depth=depth + 1))

        return lines

    lines = []
    lines.append("# Dataset Tools CLI Reference (`dst`)")
    lines.append("")
    lines.append("This reference is generated from `dataset_tools.dst.build_parser()` and documents the current CLI contract.")
    lines.append("")
    lines.append("- Primary entrypoint script: `./dst`")
    lines.append("- Python module entrypoint: `python -m dataset_tools.dst`")
    lines.append("")
    lines.extend(render_command(cli_inventory, depth=2))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate docs/dst API and CLI reference files")
    parser.add_argument("--package-root", default="src/dataset_tools")
    parser.add_argument("--docs-root", default="docs/dst")
    args = parser.parse_args()

    package_root = Path(args.package_root)
    docs_root = Path(args.docs_root)
    docs_root.mkdir(parents=True, exist_ok=True)

    api_inventory = _build_api_inventory(package_root)
    cli_inventory = _build_cli_inventory()

    (docs_root / "api-reference.md").write_text(_render_api_markdown(api_inventory), encoding="utf-8")
    (docs_root / "cli-reference.md").write_text(_render_cli_markdown(cli_inventory), encoding="utf-8")

    # Optional artifact for debugging/regeneration audits
    (docs_root / "_generated_inventory.json").write_text(
        json.dumps({"api": api_inventory, "cli": cli_inventory}, indent=2),
        encoding="utf-8",
    )

    print("Generated docs:")
    print("-", docs_root / "api-reference.md")
    print("-", docs_root / "cli-reference.md")
    print("-", docs_root / "_generated_inventory.json")


if __name__ == "__main__":
    main()
