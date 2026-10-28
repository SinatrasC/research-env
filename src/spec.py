from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from envelope import SCHEMA_VERSION

# verifiers v1 runtimes. `prime` (remote sandboxes) is billed, so a spec must name its runtime.
RUNTIME_TYPES = ("subprocess", "docker", "prime", "custom")


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class AdapterSpec:
    module: str


@dataclass(frozen=True)
class FixedResourceSpec:
    id: str
    path: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ContractsSpec:
    fixed: tuple[str, ...]
    mutable: tuple[str, ...]
    substrate_mutation: bool = False
    fixed_resources: tuple[FixedResourceSpec, ...] = ()


@dataclass(frozen=True)
class ExecutionSpec:
    artifact_root: str = "artifacts"
    default_timeout_seconds: int = 600
    arbitrary_subjects: bool = True


@dataclass(frozen=True)
class TasksetSpec:
    """A verifiers v1 taskset: its installed id and the `TasksetConfig` values to load it with."""

    id: str
    config: dict[str, Any]


@dataclass(frozen=True)
class EvalShapeSpec:
    """The evaluation shape `run-subject` holds fixed for every subject. A subject supplies the
    model (and may override the client endpoint); everything else comes from here."""

    harness: dict[str, Any]
    sampling: dict[str, Any]
    num_rollouts: int
    max_concurrent: int | None
    max_turns: int | None
    client: dict[str, Any]


@dataclass(frozen=True)
class SubstrateSpec:
    taskset: TasksetSpec
    runtime: dict[str, Any]
    eval: EvalShapeSpec


@dataclass(frozen=True)
class ResearchEnvSpec:
    path: Path
    root: Path
    schema_version: str
    id: str
    source_env: str
    adapter: AdapterSpec
    contracts: ContractsSpec
    execution: ExecutionSpec
    substrate: SubstrateSpec

    def resolve(self, path: str) -> Path:
        return (self.root / path).resolve()

    @property
    def artifact_root(self) -> Path:
        return self.resolve(self.execution.artifact_root)


def discover_spec(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        candidate = path / "research_env.yaml"
        if candidate.exists():
            return candidate
    raise SpecError("No research_env.yaml found in the current directory or its parents.")


def load_spec(path: str | Path | None = None) -> ResearchEnvSpec:
    spec_path = Path(path).resolve() if path else discover_spec()
    if not spec_path.exists():
        raise SpecError(f"Spec file does not exist: {spec_path}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SpecError("Spec must be a YAML mapping.")
    return parse_spec(raw, spec_path)


def parse_spec(raw: dict[str, Any], spec_path: Path) -> ResearchEnvSpec:
    root = spec_path.parent.resolve()
    schema_version = _required_str(raw, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SpecError(f"Unsupported schema_version: {schema_version} (this build reads {SCHEMA_VERSION}).")
    adapter_raw = _required_map(raw, "adapter")
    contracts_raw = _required_map(raw, "contracts")
    execution_raw = raw.get("execution") or {}
    fixed_resources = _fixed_resources(contracts_raw)
    spec = ResearchEnvSpec(
        path=spec_path,
        root=root,
        schema_version=schema_version,
        id=_required_str(raw, "id"),
        source_env=_required_str(raw, "source_env"),
        adapter=AdapterSpec(module=_required_str(adapter_raw, "module")),
        contracts=ContractsSpec(
            fixed=tuple(resource.id for resource in fixed_resources),
            mutable=tuple(_required_str_list(contracts_raw, "mutable")),
            substrate_mutation=bool(contracts_raw.get("substrate_mutation", False)),
            fixed_resources=tuple(fixed_resources),
        ),
        execution=ExecutionSpec(
            artifact_root=str(execution_raw.get("artifact_root", "artifacts")),
            default_timeout_seconds=int(execution_raw.get("default_timeout_seconds", 600)),
            arbitrary_subjects=bool(execution_raw.get("arbitrary_subjects", True)),
        ),
        substrate=_substrate(_required_map(raw, "substrate")),
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: ResearchEnvSpec) -> None:
    if not spec.adapter.module:
        raise SpecError("adapter.module is required.")
    if Path(spec.execution.artifact_root).is_absolute():
        raise SpecError("execution.artifact_root must be relative to the spec directory.")
    _ensure_within_root(spec.root, spec.artifact_root, "execution.artifact_root")
    for resource in spec.contracts.fixed_resources:
        if resource.path is not None:
            if Path(resource.path).is_absolute():
                raise SpecError(f"Fixed resource {resource.id}: path must be relative to the spec directory.")
            _ensure_within_root(spec.root, spec.resolve(resource.path), f"fixed resource {resource.id}")


def _substrate(raw: dict[str, Any]) -> SubstrateSpec:
    taskset_raw = _required_map(raw, "taskset", prefix="substrate")
    runtime = _required_map(raw, "runtime", prefix="substrate")
    runtime_type = runtime.get("type")
    if runtime_type not in RUNTIME_TYPES:
        raise SpecError(f"substrate.runtime.type must be one of {', '.join(RUNTIME_TYPES)}; got {runtime_type!r}.")
    eval_raw = _required_map(raw, "eval", prefix="substrate")
    harness = _required_map(eval_raw, "harness", prefix="substrate.eval")
    if not isinstance(harness.get("id"), str) or not harness["id"].strip():
        raise SpecError('substrate.eval.harness.id must be a string (quote YAML\'s null harness as "null").')
    num_rollouts = eval_raw.get("num_rollouts", 1)
    if not isinstance(num_rollouts, int) or num_rollouts < 1:
        raise SpecError("substrate.eval.num_rollouts must be a positive integer.")
    return SubstrateSpec(
        taskset=TasksetSpec(
            id=_required_str(taskset_raw, "id", prefix="substrate.taskset"),
            config=_optional_dict(taskset_raw, "config"),
        ),
        runtime=dict(runtime),
        eval=EvalShapeSpec(
            harness=dict(harness),
            sampling=_optional_dict(eval_raw, "sampling"),
            num_rollouts=num_rollouts,
            max_concurrent=_optional_int(eval_raw, "max_concurrent"),
            max_turns=_optional_int(eval_raw, "max_turns"),
            client=_optional_dict(eval_raw, "client"),
        ),
    )


def _fixed_resources(raw: dict[str, Any]) -> list[FixedResourceSpec]:
    value = raw.get("fixed")
    if not isinstance(value, list) or not value:
        raise SpecError("Missing or invalid list: fixed")
    resources: list[FixedResourceSpec] = []
    for index, item in enumerate(value):
        if isinstance(item, str) and item.strip():
            resources.append(FixedResourceSpec(id=item.strip()))
            continue
        if isinstance(item, dict):
            resource_id = item.get("id")
            path = item.get("path")
            sha256 = item.get("sha256")
            if resource_id is None:
                resource_id = path
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise SpecError(f"Invalid fixed resource at index {index}: id or path is required.")
            if path is not None and not isinstance(path, str):
                raise SpecError(f"Invalid fixed resource at index {index}: path must be a string.")
            if sha256 is not None and not isinstance(sha256, str):
                raise SpecError(f"Invalid fixed resource at index {index}: sha256 must be a string.")
            resources.append(
                FixedResourceSpec(
                    id=resource_id.strip(),
                    path=path.strip() if isinstance(path, str) and path.strip() else None,
                    sha256=sha256.strip() if isinstance(sha256, str) and sha256.strip() else None,
                )
            )
            continue
        raise SpecError(f"Invalid fixed resource at index {index}.")
    return resources


def _ensure_within_root(root: Path, path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SpecError(f"{label} must stay within the spec directory.") from exc


def _required_map(raw: dict[str, Any], key: str, *, prefix: str = "") -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise SpecError(f"Missing or invalid mapping: {prefix + '.' if prefix else ''}{key}")
    return value


def _required_str(raw: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"Missing or invalid string: {prefix + '.' if prefix else ''}{key}")
    return value


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise SpecError(f"Invalid positive integer: {key}")
    return value


def _optional_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SpecError(f"Invalid mapping: {key}")
    return value


def _required_str_list(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise SpecError(f"Missing or invalid list: {key}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise SpecError(f"Invalid string list: {key}")
    return value
