from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from envelope import SCHEMA_VERSION


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
class TaskSourceSpec:
    type: str
    path: str | None = None
    env_id: str | None = None
    env_args: dict[str, Any] = field(default_factory=dict)
    env_args_file: str | None = None
    dataset_name: str | None = None
    dataset_subset: str | None = None
    split: str = "train"
    question_key: str = "question"
    answer_key: str = "answer"
    id_key: str | None = None
    info_key: str | None = "info"


@dataclass(frozen=True)
class SubstrateSpec:
    task_source: TaskSourceSpec


@dataclass(frozen=True)
class ResearchEnvSpec:
    path: Path
    root: Path
    schema_version: str
    id: str
    source_env: str
    adapter: AdapterSpec
    contracts: ContractsSpec
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    substrate: SubstrateSpec | None = None

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
        raise SpecError(f"Unsupported schema_version: {schema_version}")

    adapter_raw = _required_map(raw, "adapter")
    contracts_raw = _required_map(raw, "contracts")
    execution_raw = raw.get("execution") or {}
    substrate_raw = raw.get("substrate") or {}
    task_source_raw = substrate_raw.get("task_source") if isinstance(substrate_raw, dict) else None

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
        substrate=SubstrateSpec(task_source=_task_source(task_source_raw)) if task_source_raw else None,
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: ResearchEnvSpec) -> None:
    if not spec.adapter.module:
        raise SpecError("adapter.module is required.")
    if Path(spec.execution.artifact_root).is_absolute():
        raise SpecError("execution.artifact_root must be relative to the spec directory.")
    _ensure_within_root(spec.root, spec.artifact_root, "execution.artifact_root")
    if spec.substrate is None:
        raise SpecError("substrate.task_source is required.")
    source = spec.substrate.task_source
    if source.type == "local_jsonl":
        if not source.path:
            raise SpecError("substrate.task_source.path is required for local_jsonl.")
        if Path(source.path).is_absolute():
            raise SpecError("substrate.task_source.path must be relative to the spec directory.")
        task_path = spec.resolve(source.path)
        _ensure_within_root(spec.root, task_path, "substrate.task_source.path")
        if not task_path.exists():
            raise SpecError(f"Local task source does not exist: {source.path}")
    elif source.type == "huggingface":
        if not source.dataset_name:
            raise SpecError("substrate.task_source.dataset_name is required for huggingface.")
    elif source.type == "verifiers":
        if source.env_args_file:
            if Path(source.env_args_file).is_absolute():
                raise SpecError("substrate.task_source.env_args_file must be relative to the spec directory.")
            env_args_path = spec.resolve(source.env_args_file)
            _ensure_within_root(spec.root, env_args_path, "substrate.task_source.env_args_file")
            if not env_args_path.exists():
                raise SpecError(f"Verifiers env args file does not exist: {source.env_args_file}")
    else:
        raise SpecError(f"Unsupported substrate.task_source.type: {source.type}")


def _task_source(raw: Any) -> TaskSourceSpec:
    if not isinstance(raw, dict):
        raise SpecError("substrate.task_source must be a mapping.")
    return TaskSourceSpec(
        type=_required_str(raw, "type"),
        path=_optional_str(raw, "path"),
        env_id=_optional_str(raw, "env_id"),
        env_args=_optional_dict(raw, "env_args"),
        env_args_file=_optional_str(raw, "env_args_file"),
        dataset_name=_optional_str(raw, "dataset_name"),
        dataset_subset=_optional_str(raw, "dataset_subset"),
        split=str(raw.get("split", "train")),
        question_key=str(raw.get("question_key", "question")),
        answer_key=str(raw.get("answer_key", "answer")),
        id_key=_optional_str(raw, "id_key"),
        info_key=_optional_str(raw, "info_key", default="info"),
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


def _required_map(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise SpecError(f"Missing or invalid mapping: {key}")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"Missing or invalid string: {key}")
    return value


def _optional_str(raw: dict[str, Any], key: str, *, default: str | None = None) -> str | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpecError(f"Invalid string: {key}")
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
