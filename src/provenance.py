from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from envelope import SCHEMA_VERSION, sha256_bytes, sha256_json
from spec import FixedResourceSpec, ResearchEnvSpec

__version__ = "0.2.0"


def build_provenance(spec: ResearchEnvSpec | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "research_env": {
            "schema_version": SCHEMA_VERSION,
            "package_version": __version__,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "process": {
            "cwd": str(Path.cwd().resolve()),
        },
        "git": _git_metadata(Path.cwd()),
    }
    if spec is not None:
        data["spec"] = {
            "id": spec.id,
            "source_env": spec.source_env,
            "path": _display_path(spec.path),
            "sha256": sha256_bytes(spec.path.read_bytes()),
            "adapter_module": spec.adapter.module,
            "schema_version": spec.schema_version,
        }
        data["substrate"] = _substrate_provenance(spec)
    return data


def build_contract_status(spec: ResearchEnvSpec | None = None) -> dict[str, Any] | None:
    if spec is None:
        return None
    resources = [_fixed_resource_status(spec, resource) for resource in spec.contracts.fixed_resources]
    changed = [resource["id"] for resource in resources if resource.get("changed") is True]
    return {
        "fixed_resources_unchanged": not changed,
        "changed_fixed_resources": changed,
        "fixed_resources": resources,
    }


def _substrate_provenance(spec: ResearchEnvSpec) -> dict[str, Any]:
    substrate = spec.substrate
    return {
        "taskset_id": substrate.taskset.id,
        "taskset_config_sha256": sha256_json(substrate.taskset.config),
        "taskset_package_version": _package_version(substrate.taskset.id),
        "verifiers_version": _package_version("verifiers"),
        "runtime_type": substrate.runtime.get("type"),
        "eval_shape_sha256": sha256_json(
            {
                "harness": substrate.eval.harness,
                "sampling": substrate.eval.sampling,
                "num_rollouts": substrate.eval.num_rollouts,
                "max_concurrent": substrate.eval.max_concurrent,
                "max_turns": substrate.eval.max_turns,
                "client": substrate.eval.client,
            }
        ),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _fixed_resource_status(spec: ResearchEnvSpec, resource: FixedResourceSpec) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": resource.id,
        "path": resource.path,
        "exists": None,
        "kind": "declared",
        "sha256": None,
        "expected_sha256": resource.sha256,
        "baseline_known": resource.sha256 is not None,
        "changed": False,
    }
    if not resource.path:
        return data

    path = spec.resolve(resource.path)
    data["exists"] = path.exists()
    if not path.exists():
        data["kind"] = "missing"
        data["changed"] = resource.sha256 is not None
        return data

    if path.is_file():
        data["kind"] = "file"
        digest = sha256_bytes(path.read_bytes())
    elif path.is_dir():
        data["kind"] = "directory"
        digest = _hash_directory(path)
    else:
        data["kind"] = "other"
        digest = None

    data["sha256"] = digest
    data["changed"] = bool(resource.sha256 and digest != resource.sha256)
    return data


def _hash_directory(path: Path) -> str:
    rows: list[dict[str, str]] = []
    # Bytecode and build metadata change on import or install, not when the resource changes.
    for child in sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
        and not any(part.endswith(".egg-info") for part in item.parts)
    ):
        rows.append(
            {
                "path": str(child.relative_to(path)).replace("\\", "/"),
                "sha256": sha256_bytes(child.read_bytes()),
            }
        )
    return sha256_json(rows)


def _git_metadata(cwd: Path) -> dict[str, Any] | None:
    def run(*args: str) -> str | None:
        try:
            process = subprocess.run(
                ["git", *args],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if process.returncode != 0:
            return None
        return process.stdout.strip()

    root = run("rev-parse", "--show-toplevel")
    if not root:
        return None
    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "root": root,
        "head": head,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode("utf-8")) if status is not None else None,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())
