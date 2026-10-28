"""research-env instruments over a verifiers v1 taskset.

`describe` and `sample-task` load the taskset in-process through `verifiers.v1`. `evaluate-candidate`
runs `vf-validate` and `run-subject` runs `vf-eval`; both receive a generated JSON config and are read
back from the run directory verifiers writes (`summary.json`/`results.jsonl`, `traces.jsonl`), never
from printed output. `compare-results` pairs two such runs task by task.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from verifiers.v1.task import Task
from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

from adapters.base import Adapter as BaseAdapter
from adapters.base import AdapterError
from envelope import InstrumentPayload, params, sha256_bytes
from modes import debug_visible, evidence_mode, references_visible
from spec import ResearchEnvSpec
from stats import paired_summary

# TaskData fields an agent may see in evaluation mode. Every other field (answers, gold actions,
# seeded state, info) can carry the reference, so it is shown only in analysis or debug mode.
PUBLIC_TASK_FIELDS = ("idx", "id", "name", "description", "prompt", "system_prompt")
VALIDATE_CHECKS = ("all", "setup", "gold", "noop")
# vf-validate's results.jsonl records task positions, not TaskData.id; this file beside it maps one to
# the other so validation runs pair by id like eval runs do.
TASK_IDS_FILE = "research_env_task_ids.json"


class Adapter(BaseAdapter):
    def __init__(self, spec: ResearchEnvSpec):
        super().__init__(spec)

    # ---- instruments ----

    def describe(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        substrate = self.spec.substrate
        result: dict[str, Any] = {
            "source_env": self.spec.source_env,
            "purpose": "Expose a verifiers v1 taskset through the research-env instruments.",
            "contracts": {
                "fixed": list(self.spec.contracts.fixed),
                "mutable": list(self.spec.contracts.mutable),
                "substrate_mutation": self.spec.contracts.substrate_mutation,
            },
            "substrate": {
                "taskset": {"id": substrate.taskset.id, "config": substrate.taskset.config},
                "runtime": substrate.runtime,
                "eval_shape": {
                    "harness": substrate.eval.harness,
                    "sampling": substrate.eval.sampling,
                    "num_rollouts": substrate.eval.num_rollouts,
                    "max_concurrent": substrate.eval.max_concurrent,
                    "max_turns": substrate.eval.max_turns,
                },
            },
            "instruments": {
                "describe": "This contract. params.probe=true also loads the taskset and reports its tasks and signals.",
                "sample-task": "Tasks from the taskset; evaluation mode shows only prompt and identity fields.",
                "evaluate-candidate": "vf-validate (setup, gold and noop checks) on a candidate taskset, optionally against a base.",
                "run-subject": "vf-eval of one subject model under the fixed evaluation shape; reads traces.jsonl.",
                "inspect-artifact": "Read an artifact under the artifact root.",
                "compare-results": "Pair two run-subject runs (rewards) or two evaluate-candidate runs (validity) task by task.",
            },
        }
        metrics: dict[str, Any] = {}
        if p.get("probe"):
            taskset = self._load_taskset(self._taskset_override(p))
            tasks = list(itertools.islice(iter(taskset), int(p.get("max_count", 10000))))
            first = tasks[0] if tasks else None
            result["probe"] = {
                "taskset_class": type(taskset).__name__,
                "bounded": taskset.bounded,
                "task_count": len(tasks) if taskset.bounded else None,
                "task_class": type(first).__name__ if first else None,
                "rewards": [fn.__name__ for fn in first.hooks("reward")] if first else [],
                "metrics": [fn.__name__ for fn in first.hooks("metric")] if first else [],
                "toolsets": [type(t).__name__ for t in type(first).toolsets(first.config)] if first else [],
                "has_gold_check": bool(first) and type(first).validate is not Task.validate,
            }
            metrics["task_count"] = len(tasks)
        return InstrumentPayload(result=result, metrics=metrics)

    def sample_task(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        mode = evidence_mode(input_data)
        taskset = self._load_taskset(self._taskset_override(p))
        limit = int(p.get("limit", 1))
        offset = int(p.get("offset", 0))
        wanted = set(p.get("task_ids") or [])
        # Infinite tasksets are generators: never scan more than max_scan tasks.
        scanned = itertools.islice(iter(taskset), int(p.get("max_scan", 10000)))
        if wanted:
            selected = [task for task in scanned if task.data.id in wanted]
        else:
            selected = list(itertools.islice(scanned, offset, offset + limit))
        tasks = [self._public_task(task, mode) for task in selected]
        return InstrumentPayload(
            result={"evidence_mode": mode, "tasks": tasks},
            metrics={"returned": len(tasks)},
            warnings=[] if references_visible(mode) else ["Evaluation mode: reference fields are hidden."],
        )

    def evaluate_candidate(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        check = str(p.get("check", "all"))
        if check not in VALIDATE_CHECKS:
            raise AdapterError("invalid_input", f"check must be one of: {', '.join(VALIDATE_CHECKS)}.")
        candidate = self._taskset_override(p.get("candidate") or {})
        candidate_run = self._validate(candidate, check, p)
        result: dict[str, Any] = {"candidate": candidate_run["result"]}
        metrics = {f"candidate_{k}": v for k, v in candidate_run["metrics"].items()}
        artifacts = list(candidate_run["artifacts"])
        evidence = [
            {
                "id": "ev_001",
                "kind": "validation",
                "label": "candidate_validity_gate",
                "value": candidate_run["metrics"]["valid_rate"],
                "confidence": 1.0,
                "supports": f"vf-validate {check} checks: {candidate_run['metrics']['valid']} of "
                f"{candidate_run['metrics']['total']} candidate tasks valid.",
                "artifact_refs": [a["id"] for a in candidate_run["artifacts"]],
            }
        ]
        if p.get("base") is not None:
            base_run = self._validate(self._taskset_override(p["base"]), check, p, prefix="base")
            result["base"] = base_run["result"]
            metrics.update({f"base_{k}": v for k, v in base_run["metrics"].items()})
            metrics["delta_valid_rate"] = candidate_run["metrics"]["valid_rate"] - base_run["metrics"]["valid_rate"]
            artifacts.extend(base_run["artifacts"])
            evidence.append(
                {
                    "id": "ev_002",
                    "kind": "comparison",
                    "label": "candidate_vs_base_validity",
                    "value": metrics["delta_valid_rate"],
                    "confidence": 1.0,
                    "supports": "Same checks on the base taskset: valid-rate change from base to candidate.",
                    "artifact_refs": [a["id"] for a in artifacts],
                }
            )
        return InstrumentPayload(
            result=result,
            metrics=metrics,
            evidence=evidence,
            artifacts=artifacts,
            warnings=[] if candidate_run["metrics"]["invalid"] == 0 else ["Some candidate tasks failed validation."],
        )

    def run_subject(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        model = p.get("model")
        if not isinstance(model, str) or not model.strip():
            raise AdapterError("invalid_input", "run_subject requires params.model (the subject).")
        taskset = self._taskset_override(p)
        shape = self.spec.substrate.eval
        run_id = self._new_run_id("eval")
        output_dir = self.spec.artifact_root / "runs"
        config = {
            "model": model,
            "num_rollouts": shape.num_rollouts,
            "output_dir": str(output_dir),
            "run": {"dir": run_id},
            "client": {**shape.client, **(p.get("client") or {})},
            "sampling": shape.sampling,
            "env": {
                "taskset": {"id": taskset["id"], **taskset["config"]},
                "agent": {
                    "harness": shape.harness,
                    "runtime": self.spec.substrate.runtime,
                    **({"max_turns": shape.max_turns} if shape.max_turns else {}),
                },
            },
            **({"max_concurrent": shape.max_concurrent} if shape.max_concurrent else {}),
        }
        dry_run = bool(p.get("dry_run", False))
        process = self._run_vf("vf-eval", config, run_id, extra=["--dry-run", "True"] if dry_run else [], p=p)
        run_dir = output_dir / run_id
        if dry_run:
            return InstrumentPayload(
                result={"run_id": run_id, "dry_run": True, "resolved": process.stdout[-20000:]},
                metrics={},
            )
        traces = run_dir / "traces.jsonl"
        if not traces.exists():
            raise AdapterError(
                "verifiers_error",
                "vf-eval finished without traces.jsonl.",
                {
                    "returncode": process.returncode,
                    "stderr": process.stderr[-4000:],
                    "run_dir": self._relative(run_dir),
                },
            )
        episodes = [json.loads(line) for line in traces.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [self._episode_row(episode) for episode in episodes]
        per_task = self._per_task_reward(rows)
        rewards = [row["reward"] for row in rows if row["reward"] is not None]
        metrics = {
            "episodes": len(rows),
            "errored_episodes": sum(1 for row in rows if row["error"]),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "tasks": len(per_task),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
        }
        artifacts = self._run_artifacts(run_dir, ["traces.jsonl", "configs/resolved/eval.json"], "eval")
        return InstrumentPayload(
            result={
                "run_id": run_id,
                "run_dir": self._relative(run_dir),
                "subject": {"model": model},
                "taskset": taskset,
                "per_task": per_task,
                "episodes": rows,
            },
            metrics=metrics,
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "evaluation",
                    "label": "subject_mean_reward",
                    "value": metrics["mean_reward"],
                    "confidence": 1.0,
                    "supports": f"{len(rows)} episodes of {model} on {taskset['id']} under the fixed evaluation shape.",
                    "artifact_refs": [a["id"] for a in artifacts],
                }
            ],
            artifacts=artifacts,
            warnings=[] if not metrics["errored_episodes"] else [f"{metrics['errored_episodes']} episodes errored."],
        )

    def inspect_artifact(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        raw_path = p.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AdapterError("invalid_input", "inspect_artifact requires path.")
        max_bytes = int(p.get("max_bytes", 65536))
        path = self._artifact_path(raw_path)
        if not path.is_file():
            raise AdapterError("not_found", f"Artifact does not exist: {raw_path}")
        data = path.read_bytes()
        return InstrumentPayload(
            result={
                "path": self._relative(path),
                "sha256": sha256_bytes(data),
                "content": data[:max_bytes].decode("utf-8", errors="replace"),
                "truncated": len(data) > max_bytes,
                "size_bytes": len(data),
            }
        )

    def compare_results(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        left_dir = self._artifact_path(str(p.get("left_run") or ""))
        right_dir = self._artifact_path(str(p.get("right_run") or ""))
        kinds = {self._run_kind(left_dir), self._run_kind(right_dir)}
        if len(kinds) != 1:
            raise AdapterError("invalid_input", "left_run and right_run must both be eval runs or both validate runs.")
        kind = kinds.pop()
        if kind == "eval":
            left = self._per_task_reward([self._episode_row(e) for e in self._jsonl(left_dir / "traces.jsonl")])
            right = self._per_task_reward([self._episode_row(e) for e in self._jsonl(right_dir / "traces.jsonl")])
            value_name = "mean_reward"
        else:
            left = {row["pair_key"]: float(row["valid"]) for row in self._validate_rows(left_dir)}
            right = {row["pair_key"]: float(row["valid"]) for row in self._validate_rows(right_dir)}
            value_name = "valid"
        common = sorted(set(left) & set(right))
        summary = paired_summary([left[k] for k in common], [right[k] for k in common])
        pairs = [{"pair_key": k, "left": left[k], "right": right[k], "delta": right[k] - left[k]} for k in common]
        return InstrumentPayload(
            result={
                "kind": kind,
                "value": value_name,
                "left_run": self._relative(left_dir),
                "right_run": self._relative(right_dir),
                "pairs": pairs,
                "unpaired_left": sorted(set(left) - set(right)),
                "unpaired_right": sorted(set(right) - set(left)),
            },
            metrics=summary,
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "comparison",
                    "label": f"paired_{value_name}",
                    "value": summary["mean_delta"],
                    "confidence": 1.0,
                    "supports": f"{summary['n_pairs']} tasks paired by TaskData.id (else task key): "
                    f"right better on {summary['right_better_count']}, worse on {summary['right_worse_count']}, "
                    f"tied on {summary['tied_count']}; sign test p={summary['sign_test_two_sided_p']}.",
                    "artifact_refs": [],
                }
            ],
            warnings=[] if len(common) == len(left) == len(right) else ["Some tasks appear in only one run."],
        )

    # ---- verifiers plumbing ----

    def _taskset_override(self, p: dict[str, Any]) -> dict[str, Any]:
        """The spec's taskset with an optional candidate `taskset: {id, config}` merged over it."""
        base = self.spec.substrate.taskset
        override = p.get("taskset") or {}
        if not isinstance(override, dict):
            raise AdapterError("invalid_input", "taskset must be a mapping with optional id and config.")
        config = {**base.config, **(override.get("config") or {})}
        return {"id": str(override.get("id") or base.id), "config": config}

    def _load_taskset(self, taskset: dict[str, Any]):
        try:
            config_type = taskset_config_type(taskset["id"])
            return load_taskset(config_type(id=taskset["id"], **taskset["config"]))
        except Exception as exc:  # the taskset package is external code: report, don't crash
            raise AdapterError(
                "taskset_load_error", f"Could not load taskset {taskset['id']}: {type(exc).__name__}: {exc}"
            ) from exc

    def _validate(self, taskset: dict[str, Any], check: str, p: dict[str, Any], prefix: str = "candidate") -> dict:
        run_id = self._new_run_id(f"validate-{prefix}")
        output_dir = self.spec.artifact_root / "validate"
        config: dict[str, Any] = {
            "taskset": {"id": taskset["id"], **taskset["config"]},
            "runtime": self.spec.substrate.runtime,
            "output_dir": str(output_dir),
            "run": {"dir": run_id},
        }
        if p.get("limit") is not None:
            config["select"] = {"limit": int(p["limit"])}
        extra = [] if check == "all" else [f"--only-{check}", "True"]
        self._run_vf("vf-validate", config, run_id, extra=extra, p=p)
        run_dir = output_dir / run_id
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise AdapterError(
                "verifiers_error", "vf-validate finished without summary.json.", {"run_dir": str(run_dir)}
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        total = int(summary.get("total", 0))
        ids = [task.data.id for task in itertools.islice(iter(self._load_taskset(taskset)), total)]
        (run_dir / TASK_IDS_FILE).write_text(json.dumps(ids) + "\n", encoding="utf-8")
        rows = self._validate_rows(run_dir)
        outcomes = summary.get("outcomes", {})
        valid = int(outcomes.get("valid", 0))
        return {
            "result": {
                "run_id": run_id,
                "run_dir": self._relative(run_dir),
                "taskset": taskset,
                "check": check,
                "checks": summary.get("checks", {}),
                "tasks": rows,
            },
            "metrics": {
                "total": total,
                "valid": valid,
                "invalid": int(outcomes.get("invalid", 0)),
                "errored": int(outcomes.get("error", 0)),
                "unchecked": int(outcomes.get("unchecked", 0)),
                "valid_rate": valid / total if total else 0.0,
            },
            "artifacts": self._run_artifacts(run_dir, ["summary.json", "results.jsonl", TASK_IDS_FILE], prefix),
        }

    def _run_vf(
        self, program: str, config: dict, run_id: str, *, extra: list[str], p: dict
    ) -> subprocess.CompletedProcess:
        executable = Path(sys.executable).parent / program
        if not executable.exists():
            raise AdapterError("missing_dependency", f"{program} is not installed next to {sys.executable}.")
        config_dir = self.spec.artifact_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{run_id}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        timeout = int(p.get("timeout_seconds", self.spec.execution.default_timeout_seconds))
        try:
            process = subprocess.run(
                [str(executable), "@", str(config_path), *extra],
                cwd=self.spec.root,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterError("timeout", f"{program} exceeded {timeout} s.") from exc
        if process.returncode != 0:
            raise AdapterError(
                "verifiers_error",
                f"{program} exited with {process.returncode}.",
                {
                    "stderr": process.stderr[-4000:],
                    "stdout": process.stdout[-2000:],
                    "config": self._relative(config_path),
                },
            )
        return process

    # ---- reading verifiers outputs ----

    def _validate_rows(self, run_dir: Path) -> list[dict[str, Any]]:
        ids_path = run_dir / TASK_IDS_FILE
        ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.is_file() else []
        rows = []
        for raw in sorted(self._jsonl(run_dir / "results.jsonl"), key=lambda r: r.get("index") or 0):
            noop = raw.get("noop") or {}
            gold = raw.get("gold") or {}
            index = raw.get("index")
            task_id = ids[index] if isinstance(index, int) and index < len(ids) else None
            rows.append(
                {
                    "pair_key": task_id or raw.get("name") or f"index_{index}",
                    "index": raw.get("index"),
                    "task_key": raw.get("task_key"),
                    "valid": bool(raw.get("valid")),
                    "reason": raw.get("reason"),
                    "error": raw.get("error"),
                    "gold": gold.get("reason"),
                    "noop": noop.get("reason"),
                    "noop_reward": noop.get("reward"),
                }
            )
        return rows

    def _episode_row(self, episode: dict[str, Any]) -> dict[str, Any]:
        """One traces.jsonl episode. The reward is the policy trace's, computed as verifiers does
        (`Trace.reward`: the sum of score x weight); it is None when the trace was not scored."""
        task = episode.get("task") or {}
        data = task.get("data") or {}
        traces = episode.get("traces") or []
        trace = traces[0] if traces else {}
        scored = [r for r in (trace.get("rewards") or {}).values() if r is not None and r.get("score") is not None]
        errors = [*(episode.get("errors") or []), *(trace.get("errors") or [])]
        agent = (trace.get("timing") or {}).get("agent") or {}
        agent_seconds = agent["end"] - agent["start"] if "start" in agent and "end" in agent else None
        return {
            "pair_key": data.get("id") or task.get("key"),
            "task_key": task.get("key"),
            "reward": sum(r["score"] * r.get("weight", 1.0) for r in scored) if scored else None,
            "rewards": {name: r.get("score") for name, r in (trace.get("rewards") or {}).items() if r is not None},
            "metrics": trace.get("metrics") or {},
            "error": errors[0] if errors else (None if episode.get("ok", True) else "episode not ok"),
            "prompt_tokens": int(episode.get("num_input_tokens") or 0),
            "completion_tokens": int(episode.get("num_output_tokens") or 0),
            "agent_seconds": agent_seconds,
        }

    def _per_task_reward(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            if row["reward"] is not None and row["pair_key"]:
                grouped.setdefault(str(row["pair_key"]), []).append(row["reward"])
        return {key: sum(values) / len(values) for key, values in sorted(grouped.items())}

    def _run_kind(self, run_dir: Path) -> str:
        if (run_dir / "traces.jsonl").is_file():
            return "eval"
        if (run_dir / "summary.json").is_file() and (run_dir / "results.jsonl").is_file():
            return "validate"
        raise AdapterError(
            "not_found", f"Not a run-subject or evaluate-candidate run directory: {self._relative(run_dir)}"
        )

    def _jsonl(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _run_artifacts(self, run_dir: Path, names: list[str], prefix: str) -> list[dict[str, Any]]:
        artifacts = []
        for index, name in enumerate(names, start=1):
            path = run_dir / name
            if path.is_file():
                artifacts.append(
                    {
                        "id": f"art_{prefix}_{index:03d}",
                        "kind": name.rsplit("/", 1)[-1],
                        "path": self._relative(path),
                        "sha256": sha256_bytes(path.read_bytes()),
                    }
                )
        return artifacts

    def _public_task(self, task: Any, mode: str) -> dict[str, Any]:
        data = task.data.model_dump(mode="json", exclude_none=True)
        out: dict[str, Any] = {"task_key": task.key}
        if references_visible(mode):
            out["data"] = data
        else:
            out["data"] = {k: v for k, v in data.items() if k in PUBLIC_TASK_FIELDS}
        if debug_visible(mode):
            out["task_class"] = type(task).__name__
            out["task_config"] = task.config.model_dump(mode="json")
        return out

    def _artifact_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise AdapterError("invalid_input", "A path under the artifact root is required.")
        path = (self.spec.artifact_root / raw_path).resolve()
        try:
            path.relative_to(self.spec.artifact_root.resolve())
        except ValueError as exc:
            raise AdapterError("invalid_input", "Paths must stay inside the artifact root.") from exc
        return path

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.spec.artifact_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(path)

    def _new_run_id(self, prefix: str) -> str:
        return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
