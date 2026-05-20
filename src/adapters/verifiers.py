from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapters.base import Adapter as BaseAdapter
from adapters.base import AdapterError
from envelope import InstrumentPayload, params, sha256_text
from modes import evidence_mode, references_visible
from spec import ResearchEnvSpec, TaskSourceSpec


@dataclass(frozen=True)
class VerifiersConfig:
    env_id: str
    env_args: dict[str, Any]
    question_key: str
    answer_key: str
    id_key: str | None
    info_key: str | None


class Adapter(BaseAdapter):
    def __init__(self, spec: ResearchEnvSpec):
        super().__init__(spec)
        if self._source().type != "verifiers":
            raise AdapterError("invalid_spec", "The Verifiers adapter requires substrate.task_source.type=verifiers.")

    def describe(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        config = self._config(p)
        result: dict[str, Any] = {
            "source_env": self.spec.source_env,
            "purpose": "Expose a Verifiers-compatible environment through the universal research-env instruments.",
            "contracts": {
                "fixed": list(self.spec.contracts.fixed),
                "mutable": list(self.spec.contracts.mutable),
                "substrate_mutation": self.spec.contracts.substrate_mutation,
            },
            "execution": {
                "artifact_root": self.spec.execution.artifact_root,
                "default_timeout_seconds": self.spec.execution.default_timeout_seconds,
                "arbitrary_subjects": self.spec.execution.arbitrary_subjects,
            },
            "task_source": {
                "type": "verifiers",
                "env_id": config.env_id,
                "question_key": config.question_key,
                "answer_key": config.answer_key,
                "id_key": config.id_key,
                "info_key": config.info_key,
            },
            "instruments": {
                "describe": {
                    "description": "Return adapter metadata. Pass params.probe=true to call vf.load_environment.",
                    "side_effects": "none",
                },
                "sample_task": {
                    "description": "Call vf.load_environment and sample public task records from env datasets.",
                    "side_effects": "none",
                },
                "evaluate_candidate": {
                    "description": "Not implemented generically; use an env-specific adapter for direct candidate scoring.",
                    "side_effects": "none",
                },
                "run_subject": {
                    "description": "Run uv run vf-eval for the configured env and save stdout/stderr/result artifacts.",
                    "side_effects": "writes_artifacts",
                },
                "inspect_artifact": {
                    "description": "Read saved Verifiers run artifacts.",
                    "side_effects": "none",
                },
                "compare_results": {
                    "description": "Compare two saved result bundles by shared numeric metrics.",
                    "side_effects": "none",
                },
            },
        }
        if bool(p.get("probe", False)):
            result["probe"] = self._verifiers_probe(config, int(p.get("timeout_seconds", 120)))
        return InstrumentPayload(result=result)

    def sample_task(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        mode = evidence_mode(input_data)
        config = self._config(p)
        payload = {
            "env_id": config.env_id,
            "env_args": config.env_args,
            "question_key": config.question_key,
            "answer_key": config.answer_key,
            "id_key": config.id_key,
            "info_key": config.info_key,
            "limit": int(p.get("limit", 1)),
            "offset": int(p.get("offset", 0)),
            "seed": p.get("seed"),
            "task_ids": p.get("task_ids"),
            "include_reference": references_visible(mode),
            "evidence_mode": mode,
        }
        sampled = self._run_verifiers_python(_SAMPLE_TASKS_SCRIPT, payload, int(p.get("timeout_seconds", 120)))
        tasks = sampled.get("tasks", [])
        return InstrumentPayload(
            result={"evidence_mode": mode, "tasks": tasks},
            metrics={"num_tasks": len(tasks)},
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "sample",
                    "label": "sampled_tasks",
                    "value": len(tasks),
                    "confidence": 1.0,
                    "supports": "Tasks were sampled from a Verifiers environment loaded with vf.load_environment.",
                    "artifact_refs": [],
                }
            ],
            warnings=sampled.get("warnings", []),
        )

    def evaluate_candidate(self, input_data: dict[str, Any]) -> InstrumentPayload:
        del input_data
        raise AdapterError(
            "not_supported",
            "Generic Verifiers candidate scoring is not reliable across envs. Use run_subject/vf-eval or an env-specific adapter.",
        )

    def run_subject(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        config = self._config(p)
        timeout_seconds = int(p.get("timeout_seconds", self.spec.execution.default_timeout_seconds))
        command = self._vf_eval_command(config, p)
        run_id = self._new_run_id()
        run_dir = self.spec.artifact_root / "verifiers_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        timed_out = False
        try:
            process = subprocess.run(
                command,
                cwd=self.spec.root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = process.returncode
            stdout = process.stdout
            stderr = process.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")

        metrics = self._extract_metrics(stdout + "\n" + stderr)
        result = {
            "run_id": run_id,
            "env_id": config.env_id,
            "command": command,
            "returncode": returncode,
            "timed_out": timed_out,
            "status": "timeout" if timed_out else ("completed" if returncode == 0 else "failed"),
        }
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        result_path = run_dir / "results.json"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result_path.write_text(
            json.dumps({"result": result, "metrics": metrics}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = [
            self._artifact("art_001", "stdout", stdout_path, "vf-eval stdout."),
            self._artifact("art_002", "stderr", stderr_path, "vf-eval stderr."),
            self._artifact("art_003", "result_bundle", result_path, "Parsed vf-eval result bundle."),
        ]
        warnings = [] if returncode == 0 else [f"vf-eval exited with status {returncode}."]
        return InstrumentPayload(
            result=result,
            metrics=metrics,
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "run",
                    "label": "vf_eval_run",
                    "value": result["status"],
                    "confidence": 1.0,
                    "supports": "The configured Verifiers environment was run with vf-eval.",
                    "artifact_refs": [artifact["id"] for artifact in artifacts],
                }
            ],
            artifacts=artifacts,
            warnings=warnings,
        )

    def inspect_artifact(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        raw_path = p.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AdapterError("invalid_input", "inspect_artifact requires path.")
        max_bytes = int(p.get("max_bytes", 65536))
        path = self._artifact_path(raw_path)
        if not path.exists() or not path.is_file():
            raise AdapterError("not_found", f"Artifact does not exist: {raw_path}")
        data = path.read_bytes()
        return InstrumentPayload(
            result={
                "path": self._relative(path),
                "content": data[:max_bytes].decode("utf-8", errors="replace"),
                "truncated": len(data) > max_bytes,
                "size_bytes": len(data),
            }
        )

    def compare_results(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        left_path = p.get("left_path") or p.get("left")
        right_path = p.get("right_path") or p.get("right")
        if not isinstance(left_path, str) or not isinstance(right_path, str):
            raise AdapterError("invalid_input", "compare_results requires left_path and right_path.")
        left_data = self._read_json_artifact(left_path)
        right_data = self._read_json_artifact(right_path)
        left_metrics = self._numeric_metrics(left_data.get("metrics", {}))
        right_metrics = self._numeric_metrics(right_data.get("metrics", {}))
        common = sorted(set(left_metrics) & set(right_metrics))
        comparisons = {
            key: {
                "left": left_metrics[key],
                "right": right_metrics[key],
                "delta": right_metrics[key] - left_metrics[key],
            }
            for key in common
        }
        return InstrumentPayload(
            result={
                "left_path": self._relative(self._artifact_path(left_path)),
                "right_path": self._relative(self._artifact_path(right_path)),
                "comparisons": comparisons,
            },
            metrics={f"delta_{key}": value["delta"] for key, value in comparisons.items()},
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "comparison",
                    "label": "shared_metric_comparison",
                    "value": comparisons,
                    "confidence": 1.0,
                    "supports": "Compared common numeric metrics from two Verifiers result bundles.",
                    "artifact_refs": [],
                }
            ],
        )

    def _config(self, overrides: dict[str, Any] | None = None) -> VerifiersConfig:
        overrides = overrides or {}
        source = self._source()
        env_args = self._env_args_from_source(source)
        if isinstance(overrides.get("env_args"), dict):
            env_args.update(overrides["env_args"])
        return VerifiersConfig(
            env_id=str(overrides.get("env_id") or source.env_id or self.spec.source_env),
            env_args=env_args,
            question_key=str(overrides.get("question_key") or source.question_key),
            answer_key=str(overrides.get("answer_key") or source.answer_key),
            id_key=overrides.get("id_key") or source.id_key,
            info_key=overrides.get("info_key") or source.info_key,
        )

    def _source(self) -> TaskSourceSpec:
        if self.spec.substrate is None:
            raise AdapterError("invalid_spec", "substrate.task_source is required.")
        return self.spec.substrate.task_source

    def _env_args_from_source(self, source: TaskSourceSpec) -> dict[str, Any]:
        env_args = dict(source.env_args)
        if source.env_args_file:
            path = self.spec.resolve(source.env_args_file)
            env_args.update(json.loads(path.read_text(encoding="utf-8")))
        return env_args

    def _verifiers_probe(self, config: VerifiersConfig, timeout_seconds: int) -> dict[str, Any]:
        return self._run_verifiers_python(
            _PROBE_SCRIPT,
            {"env_id": config.env_id, "env_args": config.env_args},
            timeout_seconds,
        )

    def _run_verifiers_python(self, script: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        try:
            process = subprocess.run(
                ["uv", "run", "python", "-c", script],
                cwd=self.spec.root,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterError("timeout", "Timed out while calling vf.load_environment.") from exc
        if process.returncode != 0:
            raise AdapterError(
                "verifiers_error",
                "Verifiers subprocess failed.",
                {"returncode": process.returncode, "stderr": process.stderr[-4000:]},
            )
        try:
            data = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError("invalid_output", "Verifiers subprocess did not return JSON.") from exc
        if not isinstance(data, dict):
            raise AdapterError("invalid_output", "Verifiers subprocess output must be a JSON object.")
        if data.get("ok") is False:
            raise AdapterError(str(data.get("kind", "verifiers_error")), str(data.get("message", "Verifiers failed.")))
        return data

    def _vf_eval_command(self, config: VerifiersConfig, p: dict[str, Any]) -> list[str]:
        command = p.get("command")
        if isinstance(command, list) and all(isinstance(item, str) for item in command):
            return command
        if command is not None:
            raise AdapterError("invalid_input", "command must be a list of strings when provided.")
        args = p.get("vf_eval_args", [])
        if isinstance(args, str):
            args = args.split()
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise AdapterError("invalid_input", "vf_eval_args must be a string or list of strings.")
        built = ["uv", "run", "vf-eval", config.env_id, *args]
        if config.env_args:
            built.extend(["-a", json.dumps(config.env_args, sort_keys=True)])
        return built

    def _extract_metrics(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            stripped = line.strip()
            summary_match = re.match(r"^([A-Za-z_][\w.]*)\s*:\s*avg\s*-\s*(-?\d+(?:\.\d+)?)", stripped)
            if summary_match:
                metrics[summary_match.group(1)] = float(summary_match.group(2))
                continue
            if not stripped.startswith("{"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            self._collect_numeric(metrics, data)
        for key in ("reward", "pass_rate", "score", "accuracy"):
            match = re.search(rf"{key}[=: ]+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
            if match and key not in metrics:
                metrics[key] = float(match.group(1))
        return metrics

    def _collect_numeric(self, metrics: dict[str, float], data: dict[str, Any], prefix: str = "") -> None:
        for key, value in data.items():
            name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                metrics[name] = float(value)
            elif isinstance(value, dict):
                self._collect_numeric(metrics, value, name)

    def _artifact_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        resolved = path.resolve() if path.is_absolute() else self.spec.resolve(raw_path)
        root = self.spec.artifact_root
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise AdapterError("forbidden", "Artifact path must stay within execution.artifact_root.") from exc
        return resolved

    def _read_json_artifact(self, raw_path: str) -> dict[str, Any]:
        path = self._artifact_path(raw_path)
        if not path.exists() or not path.is_file():
            raise AdapterError("not_found", f"Artifact does not exist: {raw_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise AdapterError("invalid_artifact", f"Artifact is not a JSON object: {raw_path}")
        return data

    def _numeric_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}

    def _artifact(self, artifact_id: str, kind: str, path: Path, description: str) -> dict[str, Any]:
        content = path.read_text(encoding="utf-8")
        return {
            "id": artifact_id,
            "kind": kind,
            "path": self._relative(path),
            "description": description,
            "sha256": sha256_text(content),
        }

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.spec.root)).replace("\\", "/")

    def _new_run_id(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{stamp}_{uuid4().hex[:8]}"


_PROBE_SCRIPT = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
import verifiers as vf

env = vf.load_environment(payload["env_id"], **payload.get("env_args", {}))
dataset = getattr(env, "dataset", None)
eval_dataset = getattr(env, "eval_dataset", None)
print(json.dumps({
    "ok": True,
    "env_class": type(env).__name__,
    "has_dataset": dataset is not None or eval_dataset is not None or hasattr(env, "get_dataset"),
    "has_rubric": hasattr(env, "rubric"),
    "is_single_turn": isinstance(env, getattr(vf, "SingleTurnEnv", object)),
}))
"""


_SAMPLE_TASKS_SCRIPT = r"""
import json
import random
import sys

payload = json.loads(sys.stdin.read())
import verifiers as vf

env = vf.load_environment(payload["env_id"], **payload.get("env_args", {}))
def resolve_attr(name):
    value = getattr(env, name, None)
    return value() if callable(value) else value

dataset = resolve_attr("dataset")
if dataset is None:
    dataset = resolve_attr("eval_dataset")
if dataset is None and hasattr(env, "get_dataset"):
    try:
        dataset = env.get_dataset()
    except ValueError as exc:
        if "dataset is not set" not in str(exc):
            raise
if dataset is None and hasattr(env, "build_dataset"):
    dataset = env.build_dataset()
if dataset is None:
    dataset = resolve_attr("tasks")
if dataset is None:
    print(json.dumps({"ok": False, "kind": "unsupported_env", "message": "Loaded env has no dataset/eval_dataset/tasks attribute."}))
    raise SystemExit(0)

length = len(dataset)
indices = list(range(length))
task_ids = payload.get("task_ids")
id_key = payload.get("id_key")
if task_ids:
    wanted = set(task_ids)
    selected = []
    for index in indices:
        item = dataset[index]
        item_id = str(item.get(id_key)) if id_key and isinstance(item, dict) and id_key in item else f"task_{index:06d}"
        if item_id in wanted:
            selected.append(index)
    indices = selected
elif payload.get("seed") is not None:
    random.Random(payload["seed"]).shuffle(indices)
else:
    offset = int(payload.get("offset", 0))
    indices = indices[offset:]
indices = indices[: int(payload.get("limit", 1))]

question_key = payload.get("question_key") or "question"
answer_key = payload.get("answer_key") or "answer"
info_key = payload.get("info_key")
include_reference = bool(payload.get("include_reference", False))
tasks = []
for index in indices:
    item = dataset[index]
    if not isinstance(item, dict):
        item = dict(item)
    task_id = str(item.get(id_key)) if id_key and id_key in item else f"task_{index:06d}"
    question = item.get(question_key, "")
    if isinstance(question, list):
        question = "\n".join(str(message.get("content", message)) if isinstance(message, dict) else str(message) for message in question)
    elif isinstance(question, dict):
        question = json.dumps(question, sort_keys=True)
    task = {
        "task_id": task_id,
        "question": str(question),
        "info": item.get(info_key, {}) if info_key else {},
    }
    if include_reference:
        task["reference_answer"] = item.get(answer_key)
    tasks.append(task)

print(json.dumps({"ok": True, "tasks": tasks}))
"""
