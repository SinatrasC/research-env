from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import uuid4

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

from adapters.base import AdapterError
from adapters.verifiers import Adapter as VerifiersAdapter
from envelope import InstrumentPayload, params, sha256_bytes
from modes import debug_visible, evidence_mode, references_visible
from spec import ResearchEnvSpec

_TOOL_DECORATOR_RE = re.compile(r"^\s*@tool\s*$", re.MULTILINE)
_TIER_RE = re.compile(r"_t\d+$")


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    task_dir: Path
    metadata: dict[str, Any]
    instruction: str
    tool_count: int
    gold_steps: int

    @property
    def family(self) -> str:
        return _TIER_RE.sub("", self.task_id)

    @property
    def tier(self) -> int:
        return int(self.metadata.get("tier", 0))


class Adapter(VerifiersAdapter):
    def __init__(self, spec: ResearchEnvSpec):
        super().__init__(spec)
        if spec.source_env != "general-agent":
            raise AdapterError("invalid_spec", "The General Agent adapter requires source_env=general-agent.")

    def describe(self, input_data: dict[str, Any]) -> InstrumentPayload:
        payload = super().describe(input_data)
        tasks_dir = self._tasks_dir(required=False)
        summaries = self._task_summaries(tasks_dir) if tasks_dir else []
        by_tier: dict[int, int] = {}
        for summary in summaries:
            by_tier[summary.tier] = by_tier.get(summary.tier, 0) + 1
        payload.result["purpose"] = (
            "Expose General Agent task metadata, tools, gold traces, verifier replay, "
            "and Verifiers harness runs as a research-env substrate."
        )
        payload.result["task_source"]["env_id"] = self._config(params(input_data)).env_id
        payload.result["instruments"]["sample_task"] = {
            "description": "Sample General Agent task handles and optional analysis/debug metadata.",
            "side_effects": "none",
        }
        payload.result["instruments"]["evaluate_candidate"] = {
            "description": "Validate and compare a candidate task evolution via gold replay and verify(db).",
            "side_effects": "writes_artifacts",
        }
        payload.result["general_agent"] = {
            "tasks_dir": self._relative_or_str(tasks_dir) if tasks_dir else None,
            "task_count": len(summaries),
            "tier_counts": {f"t{tier}": count for tier, count in sorted(by_tier.items())},
            "configured_task": self._source().env_args.get("task"),
            "task_evolution_case": "calendar_scheduling_t0 -> calendar_scheduling_capacity_t1",
            "artifact_namespace": "general_agent_evolution",
            "scientific_study_capabilities": {
                "sample_task": "Supports tier/pass-rate filters and deterministic diverse sampling.",
                "evaluate_candidate": "Supports batch study manifests for gold replay and verify(db) inclusion checks.",
                "run_subject": "Supports study manifests for paired local-solver vf-eval runs with Prime model preflight.",
                "compare_results": "Supports study bundles with paired pass-rate and complexity statistics.",
            },
        }
        return payload

    def sample_task(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        mode = evidence_mode(input_data)
        tasks_dir = self._tasks_dir(required=True)
        selected = self._select_tasks(tasks_dir, p)
        tasks = [self._public_task(summary, mode) for summary in selected]
        return InstrumentPayload(
            result={"evidence_mode": mode, "tasks": tasks},
            metrics={
                "num_tasks": len(tasks),
                "total_tools": sum(summary.tool_count for summary in selected),
                "total_gold_steps": sum(summary.gold_steps for summary in selected),
            },
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "sample",
                    "label": "general_agent_tasks",
                    "value": len(tasks),
                    "confidence": 1.0,
                    "supports": "General Agent task handles were sampled from the configured task corpus.",
                    "artifact_refs": [],
                }
            ],
        )

    def evaluate_candidate(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        mode = evidence_mode(input_data)
        if p.get("study_manifest_path"):
            return self._evaluate_study_manifest(p, mode)
        candidate_dir = self._candidate_task_dir(p)
        base_task_id = str(p.get("base_task_id") or self._source().env_args.get("task") or "")
        if not base_task_id:
            raise AdapterError("invalid_input", "evaluate_candidate requires base_task_id or substrate env_args.task.")
        base_dir = self._task_dir_by_id(base_task_id)

        base = self._evaluate_task_dir(base_dir, mode)
        candidate = self._evaluate_task_dir(candidate_dir, mode)
        comparison = self._compare_task_evidence(base, candidate)
        valid = bool(base["valid"] and candidate["valid"])
        metrics = {
            "base_valid": base["valid"],
            "candidate_valid": candidate["valid"],
            "candidate_initial_verify": candidate["initial_verify"],
            "candidate_gold_verify": candidate["gold_verify"],
            "base_tool_count": base["tool_count"],
            "candidate_tool_count": candidate["tool_count"],
            "delta_tool_count": comparison["delta_tool_count"],
            "base_gold_steps": base["gold_steps"],
            "candidate_gold_steps": candidate["gold_steps"],
            "delta_gold_steps": comparison["delta_gold_steps"],
            "instruction_length_delta": comparison["instruction_length_delta"],
        }
        run_id = self._new_evolution_id()
        run_dir = self.spec.artifact_root / "general_agent_evolution" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "result": {
                "run_id": run_id,
                "evidence_mode": mode,
                "base_task_id": base["task_id"],
                "candidate_task_id": candidate["task_id"],
                "candidate_task_dir": self._relative(candidate_dir),
                "valid": valid,
                "comparison": comparison,
            },
            "metrics": metrics,
            "base": base,
            "candidate": candidate,
        }
        bundle_path = run_dir / "task_evolution_evaluation.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts = [
            self._artifact(
                "art_001",
                "task_evolution_bundle",
                bundle_path,
                "General Agent task-evolution validation and comparison bundle.",
            )
        ]
        return InstrumentPayload(
            result=bundle["result"],
            metrics=metrics,
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "validation",
                    "label": "candidate_gold_replay_and_verify",
                    "value": candidate["valid"],
                    "confidence": 1.0,
                    "supports": "Candidate gold trace changes DB state and verify(db) accepts the gold DB."
                    if candidate["valid"]
                    else "Candidate failed at least one gold replay or verify(db) validation check.",
                    "artifact_refs": ["art_001"],
                },
                {
                    "id": "ev_002",
                    "kind": "comparison",
                    "label": "task_complexity_delta",
                    "value": comparison,
                    "confidence": 1.0,
                    "supports": "Candidate task metadata and replay evidence were compared against the base task.",
                    "artifact_refs": ["art_001"],
                },
            ],
            artifacts=artifacts,
            warnings=[] if valid else ["Base or candidate task validation failed."],
        )

    def run_subject(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        if p.get("study_manifest_path"):
            return self._run_study_manifest(p)
        return super().run_subject(input_data)

    def compare_results(self, input_data: dict[str, Any]) -> InstrumentPayload:
        p = params(input_data)
        if p.get("study_bundle_path"):
            return self._compare_study_results(p)
        return super().compare_results(input_data)

    def _public_task(self, summary: TaskSummary, mode: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "task_name": summary.task_id,
            "family": summary.family,
            "tier": summary.tier,
            "description": summary.metadata.get("description"),
            "tool_count": summary.tool_count,
            "gold_steps": summary.gold_steps,
            "task_dir": self._relative_or_str(summary.task_dir),
        }
        if references_visible(mode):
            info["pass_rates"] = summary.metadata.get("pass_rates", [])
            info["difficulty_methods"] = summary.metadata.get("difficulty_methods", [])
            info["tool_names"] = self._tool_names(summary.task_dir)
        if debug_visible(mode):
            info["gold"] = self._load_json(summary.task_dir / "gold.json", default=[])
            info["db_sha256"] = self._file_sha256(summary.task_dir / "db.json")
            info["tools_sha256"] = self._file_sha256(summary.task_dir / "tools.py")
        return {
            "task_id": summary.task_id,
            "question": summary.instruction,
            "info": info,
        }

    def _select_tasks(self, tasks_dir: Path, p: dict[str, Any]) -> list[TaskSummary]:
        summaries = self._task_summaries(tasks_dir)
        task_ids = p.get("task_ids")
        p_env_args = p.get("env_args") if isinstance(p.get("env_args"), dict) else {}
        has_filter = any(
            key in p
            for key in (
                "tier",
                "min_tier",
                "max_tier",
                "min_recorded_pass_rate",
                "max_recorded_pass_rate",
                "pass_rate_solver",
                "pass_rate_model",
                "require_recorded_pass_rate",
                "domain_diversity",
                "study_sample",
            )
        )
        single_task = p.get("task") or p_env_args.get("task")
        if single_task is None and not has_filter:
            single_task = self._source().env_args.get("task")
        family = p.get("family")
        if task_ids is not None:
            if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
                raise AdapterError("invalid_input", "task_ids must be a list of strings.")
            wanted = set(task_ids)
            selected = [summary for summary in summaries if summary.task_id in wanted]
            missing = sorted(wanted - {summary.task_id for summary in selected})
            if missing:
                raise AdapterError("invalid_input", "Unknown General Agent task_id.", {"missing": missing})
            return selected
        if isinstance(single_task, str) and single_task.strip():
            return [summary for summary in summaries if summary.task_id == single_task]
        if isinstance(family, str) and family.strip():
            summaries = [summary for summary in summaries if summary.family == family]

        if "tier" in p:
            tier = int(p["tier"])
            summaries = [summary for summary in summaries if summary.tier == tier]
        if "min_tier" in p:
            min_tier = int(p["min_tier"])
            summaries = [summary for summary in summaries if summary.tier >= min_tier]
        if "max_tier" in p:
            max_tier = int(p["max_tier"])
            summaries = [summary for summary in summaries if summary.tier <= max_tier]

        rate_solver = str(p.get("pass_rate_solver", "local"))
        rate_model = p.get("pass_rate_model")
        require_rate = bool(p.get("require_recorded_pass_rate", False))
        min_rate = p.get("min_recorded_pass_rate")
        max_rate = p.get("max_recorded_pass_rate")
        if require_rate or min_rate is not None or max_rate is not None or rate_model is not None:
            filtered = []
            for summary in summaries:
                rate = self._recorded_pass_rate(summary.metadata, solver=rate_solver, model=rate_model)
                if rate is None:
                    if require_rate:
                        continue
                    filtered.append(summary)
                    continue
                if min_rate is not None and rate < float(min_rate):
                    continue
                if max_rate is not None and rate > float(max_rate):
                    continue
                filtered.append(summary)
            summaries = filtered

        if bool(p.get("domain_diversity", False)):
            seen = set()
            diverse = []
            for summary in summaries:
                if summary.family in seen:
                    continue
                seen.add(summary.family)
                diverse.append(summary)
            summaries = diverse

        limit = int(p.get("limit", 1))
        offset = int(p.get("offset", 0))
        if limit < 0 or offset < 0:
            raise AdapterError("invalid_input", "limit and offset must be non-negative.")
        seed = p.get("seed")
        if seed is not None:
            shuffled = list(summaries)
            random.Random(seed).shuffle(shuffled)
            return shuffled[:limit]
        return summaries[offset : offset + limit]

    def _recorded_pass_rate(self, metadata: dict[str, Any], *, solver: str, model: Any = None) -> float | None:
        rows = metadata.get("pass_rates", [])
        if not isinstance(rows, list):
            return None
        matches = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("solver")) != solver:
                continue
            if model is not None and str(row.get("model")) != str(model):
                continue
            value = row.get("value")
            if isinstance(value, int | float):
                matches.append(float(value))
        if not matches:
            return None
        return matches[0]

    def _task_summaries(self, tasks_dir: Path) -> list[TaskSummary]:
        summaries = []
        for task_dir in sorted(item for item in tasks_dir.iterdir() if item.is_dir()):
            if not (task_dir / "task.toml").exists():
                continue
            metadata = self._task_metadata(task_dir)
            instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
            summaries.append(
                TaskSummary(
                    task_id=task_dir.name,
                    task_dir=task_dir,
                    metadata=metadata,
                    instruction=instruction,
                    tool_count=len(_TOOL_DECORATOR_RE.findall((task_dir / "tools.py").read_text(encoding="utf-8"))),
                    gold_steps=len(self._load_json(task_dir / "gold.json", default=[])),
                )
            )
        return summaries

    def _evaluate_task_dir(self, task_dir: Path, mode: str) -> dict[str, Any]:
        for name in ("task.toml", "db.json", "tools.py", "gold.json", "instruction.md"):
            if not (task_dir / name).exists():
                raise AdapterError("invalid_input", f"Task directory is missing {name}: {self._relative_or_str(task_dir)}")
        metadata = self._task_metadata(task_dir)
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        gold = self._load_json(task_dir / "gold.json", default=[])
        db_cls = self._load_attr(task_dir / "tools.py", "TaskDB")
        tools_cls = self._load_attr(task_dir / "tools.py", "TaskTools")
        verify_fn = self._load_attr(task_dir / "tools.py", "verify")
        if db_cls is None or tools_cls is None or verify_fn is None:
            raise AdapterError("invalid_task", "tools.py must define TaskDB, TaskTools, and verify.")

        initial_tools = tools_cls(db_cls.load(task_dir / "db.json"))
        gold_tools = tools_cls(db_cls.load(task_dir / "db.json"))
        replay_error = None
        replay_outputs = []
        for tool_name, kwargs in gold:
            try:
                output = gold_tools.call_tool(tool_name, **kwargs)
            except Exception as exc:  # intentional: replay failures are evidence
                replay_error = f"{type(exc).__name__}: {exc}"
                break
            replay_outputs.append({"tool": tool_name, "output": output})

        initial_hash = initial_tools.db.get_hash()
        gold_hash = gold_tools.db.get_hash()
        changed_db = initial_hash != gold_hash
        initial_verify = self._verify_score(verify_fn, initial_tools.db)
        gold_verify = self._verify_score(verify_fn, gold_tools.db)
        valid = replay_error is None and changed_db and initial_verify == 0.0 and gold_verify == 1.0
        tool_names = self._tool_names(task_dir)
        data: dict[str, Any] = {
            "task_id": task_dir.name,
            "task_dir": self._relative_or_str(task_dir),
            "metadata": {
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "tier": metadata.get("tier", 0),
                "parent": metadata.get("parent"),
                "difficulty_methods": metadata.get("difficulty_methods", []),
                "pass_rates": metadata.get("pass_rates", []),
            },
            "instruction_length": len(instruction),
            "tool_count": len(tool_names),
            "tool_names": tool_names,
            "gold_steps": len(gold),
            "initial_db_hash": initial_hash,
            "gold_db_hash": gold_hash,
            "gold_changes_db": changed_db,
            "initial_verify": initial_verify,
            "gold_verify": gold_verify,
            "replay_error": replay_error,
            "valid": valid,
            "file_hashes": {
                "task.toml": self._file_sha256(task_dir / "task.toml"),
                "db.json": self._file_sha256(task_dir / "db.json"),
                "tools.py": self._file_sha256(task_dir / "tools.py"),
                "gold.json": self._file_sha256(task_dir / "gold.json"),
                "instruction.md": self._file_sha256(task_dir / "instruction.md"),
            },
        }
        if references_visible(mode):
            data["instruction"] = instruction
        if debug_visible(mode):
            data["gold"] = gold
            data["replay_outputs"] = replay_outputs
            data["final_db"] = gold_tools.db.model_dump()
        return data

    def _compare_task_evidence(self, base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        base_methods = set(base["metadata"].get("difficulty_methods", []))
        candidate_methods = set(candidate["metadata"].get("difficulty_methods", []))
        return {
            "base_task_id": base["task_id"],
            "candidate_task_id": candidate["task_id"],
            "base_tier": base["metadata"].get("tier", 0),
            "candidate_tier": candidate["metadata"].get("tier", 0),
            "delta_tool_count": candidate["tool_count"] - base["tool_count"],
            "delta_gold_steps": candidate["gold_steps"] - base["gold_steps"],
            "instruction_length_delta": candidate["instruction_length"] - base["instruction_length"],
            "added_tools": sorted(set(candidate["tool_names"]) - set(base["tool_names"])),
            "removed_tools": sorted(set(base["tool_names"]) - set(candidate["tool_names"])),
            "added_difficulty_methods": sorted(candidate_methods - base_methods),
            "base_valid": base["valid"],
            "candidate_valid": candidate["valid"],
        }

    def _validate_evolution_metadata(
        self,
        candidate_dir: Path,
        base: dict[str, Any],
        candidate: dict[str, Any],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        metadata_path = candidate_dir / "evolution_metadata.json"
        if not metadata_path.exists():
            return {"valid": True, "error": None, "metadata": None}

        metadata = self._load_json(metadata_path, default={})
        if not isinstance(metadata, dict):
            return {"valid": False, "error": "evolution_metadata_not_object", "metadata": None}

        required = (
            "mutation_kind",
            "added_entity",
            "added_lookup_tool",
            "constrained_action_tool",
            "constraint_rule",
            "base_gold_steps",
            "candidate_gold_steps",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            return {
                "valid": False,
                "error": f"evolution_metadata_missing:{','.join(missing)}",
                "metadata": metadata,
            }

        added_tools = comparison.get("added_tools", [])
        if not isinstance(added_tools, list):
            added_tools = []
        added_lookup_tool = metadata.get("added_lookup_tool")
        constrained_action_tool = metadata.get("constrained_action_tool")
        candidate_gold = self._load_json(candidate_dir / "gold.json", default=[])
        candidate_gold_tools = [step[0] for step in candidate_gold if isinstance(step, list | tuple) and step]

        errors = []
        if metadata.get("mutation_kind") != "cross_entity_constraint":
            errors.append("mutation_kind_not_cross_entity_constraint")
        if len(added_tools) != 1:
            errors.append("added_tool_count_not_one")
        if added_lookup_tool not in added_tools:
            errors.append("added_lookup_tool_mismatch")
        if comparison.get("delta_tool_count") != 1:
            errors.append("delta_tool_count_not_one")
        if comparison.get("delta_gold_steps") != 1:
            errors.append("delta_gold_steps_not_one")
        if metadata.get("base_gold_steps") != base.get("gold_steps"):
            errors.append("base_gold_steps_mismatch")
        if metadata.get("candidate_gold_steps") != candidate.get("gold_steps"):
            errors.append("candidate_gold_steps_mismatch")
        if constrained_action_tool not in candidate.get("tool_names", []):
            errors.append("constrained_action_tool_missing")
        if added_lookup_tool not in candidate_gold_tools:
            errors.append("added_lookup_tool_not_in_gold")
        if constrained_action_tool not in candidate_gold_tools:
            errors.append("constrained_action_tool_not_in_gold")
        if added_lookup_tool in candidate_gold_tools and constrained_action_tool in candidate_gold_tools:
            if candidate_gold_tools.index(added_lookup_tool) > candidate_gold_tools.index(constrained_action_tool):
                errors.append("lookup_after_constrained_action")

        return {
            "valid": not errors,
            "error": None if not errors else "evolution_metadata_invalid:" + ",".join(errors),
            "metadata": metadata,
        }

    def _evaluate_study_manifest(self, p: dict[str, Any], mode: str) -> InstrumentPayload:
        manifest_path = self._manifest_path(p)
        manifest = self._load_manifest(manifest_path)
        pairs = self._manifest_pairs(manifest)
        min_included = int(p.get("min_included", 24))
        run_id = self._new_study_run_id("validation")
        run_dir = self.spec.artifact_root / "general_agent_studies" / self._manifest_id(manifest) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for pair in pairs:
            pair_id = str(pair.get("pair_id") or pair.get("id") or pair.get("candidate_task_id"))
            try:
                base_task_id = str(pair["base_task_id"])
                candidate_dir = self._pair_candidate_dir(pair, manifest_path)
                base = self._evaluate_task_dir(self._task_dir_by_id(base_task_id), mode)
                candidate = self._evaluate_task_dir(candidate_dir, mode)
                comparison = self._compare_task_evidence(base, candidate)
                evolution_metadata = self._validate_evolution_metadata(candidate_dir, base, candidate, comparison)
                included = bool(base["valid"] and candidate["valid"] and evolution_metadata["valid"])
                record = {
                    "pair_id": pair_id,
                    "base_task_id": base_task_id,
                    "candidate_task_id": candidate["task_id"],
                    "candidate_task_dir": self._relative(candidate_dir),
                    "included": included,
                    "exclusion_reason": None
                    if included
                    else evolution_metadata["error"] or "base_or_candidate_validation_failed",
                    "mutation_kind": pair.get("mutation_kind"),
                    "atomic_change": pair.get("atomic_change"),
                    "evolution_metadata": evolution_metadata["metadata"],
                    "metrics": {
                        "base_valid": base["valid"],
                        "candidate_valid": candidate["valid"],
                        "candidate_gold_replay_success": candidate["replay_error"] is None,
                        "candidate_gold_changes_db": candidate["gold_changes_db"],
                        "candidate_initial_verify": candidate["initial_verify"],
                        "candidate_gold_verify": candidate["gold_verify"],
                        "base_tool_count": base["tool_count"],
                        "candidate_tool_count": candidate["tool_count"],
                        "delta_tool_count": comparison["delta_tool_count"],
                        "base_gold_steps": base["gold_steps"],
                        "candidate_gold_steps": candidate["gold_steps"],
                        "delta_gold_steps": comparison["delta_gold_steps"],
                        "instruction_length_delta": comparison["instruction_length_delta"],
                        "evolution_metadata_valid": evolution_metadata["valid"],
                    },
                    "comparison": comparison,
                }
                if debug_visible(mode):
                    record["base"] = base
                    record["candidate"] = candidate
            except Exception as exc:  # intentional: failed candidates are evidence
                record = {
                    "pair_id": pair_id,
                    "base_task_id": pair.get("base_task_id"),
                    "candidate_task_id": pair.get("candidate_task_id"),
                    "included": False,
                    "exclusion_reason": f"{type(exc).__name__}: {exc}",
                    "mutation_kind": pair.get("mutation_kind"),
                    "atomic_change": pair.get("atomic_change"),
                    "metrics": {},
                }
            records.append(record)

        stats = self._study_statistics(records)
        result = {
            "run_id": run_id,
            "study_id": self._manifest_id(manifest),
            "manifest_path": self._relative(manifest_path),
            "evidence_mode": mode,
            "n_pairs": len(records),
            "n_included": stats["n_included"],
            "n_excluded": stats["n_excluded"],
            "min_included": min_included,
            "primary_result_ready": stats["n_included"] >= min_included,
            "records": records,
            "statistics": stats,
        }
        bundle_path = run_dir / "validation_bundle.json"
        csv_path = run_dir / "validation_summary.csv"
        bundle_path.write_text(json.dumps({"result": result, "metrics": stats}, indent=2, sort_keys=True) + "\n")
        self._write_validation_csv(csv_path, records)
        artifacts = [
            self._artifact("art_001", "study_validation_bundle", bundle_path, "Batch validation results."),
            self._artifact("art_002", "study_validation_csv", csv_path, "Batch validation summary table."),
        ]
        evidence = [
            {
                "id": "ev_001",
                "kind": "validation",
                "label": "study_inclusion_count",
                "value": stats["n_included"],
                "confidence": 1.0,
                "supports": f"{stats['n_included']}/{len(records)} candidate variants passed deterministic inclusion checks.",
                "artifact_refs": ["art_001", "art_002"],
            }
        ]
        for index, record in enumerate(records, start=1):
            if not record.get("included"):
                continue
            metrics = record.get("metrics", {})
            evidence.append(
                {
                    "id": f"ev_pair_{index:03d}_validity",
                    "kind": "validation",
                    "label": "candidate_gold_replay_verify_validity",
                    "value": {
                        "pair_id": record["pair_id"],
                        "gold_replay_success": metrics.get("candidate_gold_replay_success"),
                        "db_state_changed_by_gold": metrics.get("candidate_gold_changes_db"),
                        "verify_initial_db": metrics.get("candidate_initial_verify"),
                        "verify_gold_db": metrics.get("candidate_gold_verify"),
                    },
                    "confidence": 1.0,
                    "supports": "Candidate passed gold replay, DB-change, initial verifier, and gold verifier checks.",
                    "artifact_refs": ["art_001"],
                }
            )
        return InstrumentPayload(
            result=result,
            metrics=stats,
            evidence=evidence,
            artifacts=artifacts,
            warnings=[] if stats["n_included"] >= min_included else ["Fewer than min_included candidates passed validation."],
        )

    def _run_study_manifest(self, p: dict[str, Any]) -> InstrumentPayload:
        manifest_path = self._manifest_path(p)
        manifest = self._load_manifest(manifest_path)
        validation_bundle = self._validation_bundle_from_params(p)
        included_ids = self._included_pair_ids(validation_bundle)
        validation_records = {
            str(record.get("pair_id")): record
            for record in validation_bundle.get("result", {}).get("records", [])
            if isinstance(record, dict)
        }
        pairs = [pair for pair in self._manifest_pairs(manifest) if str(pair.get("pair_id")) in included_ids]
        if not pairs:
            raise AdapterError("invalid_input", "No included pairs found. Run evaluate-candidate study validation first.")

        dry_run = bool(p.get("dry_run", False))
        rollouts = int(p.get("rollouts_per_condition", manifest.get("run_args", {}).get("rollouts_per_condition", 3)))
        timeout_seconds = int(p.get("timeout_seconds", self.spec.execution.default_timeout_seconds))
        model_id = p.get("model_id") or manifest.get("model_selection", {}).get("selected_model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            if dry_run:
                model_id = "<prime-kimi-or-glm-model-id>"
            else:
                model_id = self._select_prime_model()
                self._record_selected_model(manifest_path, model_id)

        run_id = self._new_study_run_id("vf_eval")
        run_dir = self.spec.artifact_root / "general_agent_studies" / self._manifest_id(manifest) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        commands = []
        results = []
        for pair in pairs:
            for condition in ("base", "evolved"):
                command, job_dir = self._study_vf_eval_command(
                    pair=pair,
                    manifest_path=manifest_path,
                    condition=condition,
                    model_id=model_id,
                    rollouts=rollouts,
                    run_dir=run_dir,
                )
                commands.append({"pair_id": pair["pair_id"], "condition": condition, "command": command})
                if dry_run:
                    continue
                job_dir.mkdir(parents=True, exist_ok=True)
                process = subprocess.run(
                    command,
                    cwd=self.spec.root,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                stdout_path = job_dir / "stdout.log"
                stderr_path = job_dir / "stderr.log"
                stdout_path.write_text(process.stdout)
                stderr_path.write_text(process.stderr)
                metrics = self._extract_metrics(process.stdout + "\n" + process.stderr)
                pass_rate = self._pass_rate_from_eval_text(process.stdout + "\n" + process.stderr, metrics)
                results.append(
                    {
                        "pair_id": pair["pair_id"],
                        "condition": condition,
                        "returncode": process.returncode,
                        "status": "completed" if process.returncode == 0 else "failed",
                        "pass_rate": pass_rate,
                        "metrics": metrics,
                        "stdout_path": self._relative(stdout_path),
                        "stderr_path": self._relative(stderr_path),
                    }
                )

        pair_results = self._merge_condition_results(pairs, results, validation_records)
        stats = self._study_statistics(pair_results)
        result = {
            "run_id": run_id,
            "study_id": self._manifest_id(manifest),
            "manifest_path": self._relative(manifest_path),
            "model_id": model_id,
            "rollouts_per_condition": rollouts,
            "dry_run": dry_run,
            "commands": commands,
            "pair_results": pair_results,
            "statistics": stats,
        }
        bundle_path = run_dir / ("vf_eval_dry_run_bundle.json" if dry_run else "vf_eval_bundle.json")
        bundle_path.write_text(json.dumps({"result": result, "metrics": stats}, indent=2, sort_keys=True) + "\n")
        artifacts = [self._artifact("art_001", "study_vf_eval_bundle", bundle_path, "Paired local-solver eval study bundle.")]
        return InstrumentPayload(
            result=result,
            metrics=stats,
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "run",
                    "label": "paired_local_solver_study",
                    "value": "dry_run" if dry_run else "completed",
                    "confidence": 1.0,
                    "supports": "Built or executed paired vf-eval commands for included General Agent candidate variants.",
                    "artifact_refs": ["art_001"],
                }
            ],
            artifacts=artifacts,
        )

    def _compare_study_results(self, p: dict[str, Any]) -> InstrumentPayload:
        bundle = self._read_json_artifact(str(p["study_bundle_path"]))
        result = bundle.get("result", {})
        pair_results = result.get("pair_results") or result.get("records") or []
        if not isinstance(pair_results, list):
            raise AdapterError("invalid_artifact", "Study bundle result must contain pair_results or records.")
        stats = self._study_statistics(pair_results)
        return InstrumentPayload(
            result={
                "study_bundle_path": self._relative(self._artifact_path(str(p["study_bundle_path"]))),
                "statistics": stats,
                "primary_result_available": stats.get("n_pass_rate_pairs", 0) > 0,
            },
            metrics=stats,
            evidence=[
                {
                    "id": "ev_001",
                    "kind": "comparison",
                    "label": "study_paired_statistics",
                    "value": stats,
                    "confidence": 1.0,
                    "supports": "Computed paired pass-rate and task-complexity statistics from a study bundle.",
                    "artifact_refs": [],
                }
            ],
        )

    def _manifest_path(self, p: dict[str, Any]) -> Path:
        raw = p.get("study_manifest_path")
        if not isinstance(raw, str) or not raw.strip():
            raise AdapterError("invalid_input", "study_manifest_path is required.")
        path = Path(raw)
        resolved = path.resolve() if path.is_absolute() else self.spec.resolve(raw)
        self._ensure_inside(self.spec.root, resolved, "study_manifest_path")
        if not resolved.exists() or not resolved.is_file():
            raise AdapterError("not_found", f"Study manifest does not exist: {raw}")
        return resolved

    def _load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AdapterError("invalid_input", f"Study manifest is not valid JSON: {self._relative(manifest_path)}") from exc
        if not isinstance(manifest, dict):
            raise AdapterError("invalid_input", "Study manifest must be a JSON object.")
        self._manifest_pairs(manifest)
        return manifest

    def _manifest_pairs(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        pairs = manifest.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise AdapterError("invalid_input", "Study manifest requires a non-empty pairs list.")
        validated = []
        for index, pair in enumerate(pairs, start=1):
            if not isinstance(pair, dict):
                raise AdapterError("invalid_input", f"Study manifest pair {index} must be an object.")
            if not isinstance(pair.get("base_task_id"), str) or not pair["base_task_id"].strip():
                raise AdapterError("invalid_input", f"Study manifest pair {index} requires base_task_id.")
            has_candidate_dir = isinstance(pair.get("candidate_task_dir"), str) and bool(pair["candidate_task_dir"].strip())
            has_candidate_id = isinstance(pair.get("candidate_task_id"), str) and bool(pair["candidate_task_id"].strip())
            if not has_candidate_dir and not has_candidate_id:
                raise AdapterError(
                    "invalid_input",
                    f"Study manifest pair {index} requires candidate_task_dir or candidate_task_id.",
                )
            if "pair_id" not in pair:
                pair = dict(pair)
                pair["pair_id"] = f"pair_{index:03d}"
            validated.append(pair)
        return validated

    def _manifest_id(self, manifest: dict[str, Any]) -> str:
        raw = manifest.get("id") or manifest.get("study_id") or "general_agent_study"
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._-")
        return text or "general_agent_study"

    def _pair_candidate_dir(self, pair: dict[str, Any], manifest_path: Path) -> Path:
        raw = pair.get("candidate_task_dir")
        if isinstance(raw, str) and raw.strip():
            path = Path(raw)
            resolved = path.resolve() if path.is_absolute() else (manifest_path.parent / raw).resolve()
        else:
            resolved = (manifest_path.parent / "candidate_tasks" / str(pair["candidate_task_id"])).resolve()
        self._ensure_inside(self.spec.root, resolved, "candidate_task_dir")
        return resolved

    def _study_statistics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        included_rows = [row for row in rows if bool(row.get("included", True))]
        excluded_rows = [row for row in rows if not bool(row.get("included", True))]
        pass_rate_rows = []
        for row in included_rows:
            base_rate = self._row_pass_rate(row, "base")
            evolved_rate = self._row_pass_rate(row, "evolved")
            if base_rate is None or evolved_rate is None:
                continue
            pass_rate_rows.append((base_rate, evolved_rate))

        drops = [base - evolved for base, evolved in pass_rate_rows]
        positive = sum(1 for drop in drops if drop > 0)
        negative = sum(1 for drop in drops if drop < 0)
        ties = sum(1 for drop in drops if drop == 0)
        complexity = self._complexity_delta_stats(included_rows)
        return {
            "n_pairs": len(rows),
            "n_included": len(included_rows),
            "n_excluded": len(excluded_rows),
            "n_pass_rate_pairs": len(pass_rate_rows),
            "base_mean_pass_rate": self._safe_mean([base for base, _ in pass_rate_rows]),
            "evolved_mean_pass_rate": self._safe_mean([evolved for _, evolved in pass_rate_rows]),
            "mean_paired_pass_rate_drop": self._safe_mean(drops),
            "median_paired_pass_rate_drop": median(drops) if drops else None,
            "evolved_lt_base_count": positive,
            "evolved_eq_base_count": ties,
            "evolved_gt_base_count": negative,
            "sign_test_non_tie_n": positive + negative,
            "sign_test_two_sided_p": self._sign_test_two_sided(positive, negative),
            "bootstrap_mean_drop_ci95": self._bootstrap_ci95(drops),
            **complexity,
        }

    def _row_pass_rate(self, row: dict[str, Any], condition: str) -> float | None:
        direct = row.get(f"{condition}_pass_rate")
        if isinstance(direct, int | float):
            return float(direct)
        nested = row.get(condition)
        if isinstance(nested, dict) and isinstance(nested.get("pass_rate"), int | float):
            return float(nested["pass_rate"])
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get(f"{condition}_pass_rate")
            if isinstance(value, int | float):
                return float(value)
        return None

    def _complexity_delta_stats(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        deltas: dict[str, list[float]] = {
            "delta_tool_count": [],
            "delta_gold_steps": [],
            "instruction_length_delta": [],
        }
        for row in rows:
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for key in deltas:
                value = metrics.get(key)
                if isinstance(value, int | float):
                    deltas[key].append(float(value))
        return {
            "mean_delta_tool_count": self._safe_mean(deltas["delta_tool_count"]),
            "mean_delta_gold_steps": self._safe_mean(deltas["delta_gold_steps"]),
            "mean_instruction_length_delta": self._safe_mean(deltas["instruction_length_delta"]),
        }

    def _safe_mean(self, values: list[float]) -> float | None:
        return mean(values) if values else None

    def _bootstrap_ci95(self, values: list[float]) -> list[float] | None:
        if not values:
            return None
        if len(values) == 1:
            return [values[0], values[0]]
        rng = random.Random(20260520)
        boot = []
        for _ in range(2000):
            sample = [values[rng.randrange(len(values))] for _ in values]
            boot.append(mean(sample))
        boot.sort()
        return [boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]]

    def _sign_test_two_sided(self, positive: int, negative: int) -> float | None:
        n = positive + negative
        if n == 0:
            return None
        tail = min(positive, negative)
        probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
        return min(1.0, 2.0 * probability)

    def _write_validation_csv(self, path: Path, records: list[dict[str, Any]]) -> None:
        columns = [
            "pair_id",
            "base_task_id",
            "candidate_task_id",
            "included",
            "exclusion_reason",
            "mutation_kind",
            "atomic_change",
            "candidate_gold_replay_success",
            "candidate_gold_changes_db",
            "candidate_initial_verify",
            "candidate_gold_verify",
            "delta_tool_count",
            "delta_gold_steps",
            "instruction_length_delta",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                metrics = record.get("metrics", {})
                row = {key: record.get(key) for key in columns}
                if isinstance(metrics, dict):
                    row.update({key: metrics.get(key) for key in columns if key in metrics})
                writer.writerow(row)

    def _new_study_run_id(self, prefix: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "study"
        return f"{safe_prefix}_{stamp}_{uuid4().hex[:8]}"

    def _validation_bundle_from_params(self, p: dict[str, Any]) -> dict[str, Any]:
        bundle = p.get("validation_bundle")
        if isinstance(bundle, dict):
            return bundle
        raw_path = p.get("validation_bundle_path")
        if isinstance(raw_path, str) and raw_path.strip():
            return self._read_json_artifact(raw_path)
        raise AdapterError(
            "invalid_input",
            "run-subject study mode requires validation_bundle_path from evaluate-candidate.",
        )

    def _included_pair_ids(self, validation_bundle: dict[str, Any]) -> set[str]:
        records = validation_bundle.get("result", {}).get("records", [])
        if not isinstance(records, list):
            raise AdapterError("invalid_artifact", "Validation bundle result.records must be a list.")
        return {str(record.get("pair_id")) for record in records if isinstance(record, dict) and record.get("included")}

    def _study_vf_eval_command(
        self,
        *,
        pair: dict[str, Any],
        manifest_path: Path,
        condition: str,
        model_id: str,
        rollouts: int,
        run_dir: Path,
    ) -> tuple[list[str], Path]:
        if condition == "base":
            task_id = str(pair["base_task_id"])
            tasks_dir = self._tasks_dir(required=True)
        elif condition == "evolved":
            task_id = str(pair.get("candidate_task_id") or self._pair_candidate_dir(pair, manifest_path).name)
            tasks_dir = self._pair_candidate_dir(pair, manifest_path).parent
        else:
            raise AdapterError("invalid_input", f"Unknown study condition: {condition}")
        if tasks_dir is None:
            raise AdapterError("invalid_spec", "Could not resolve tasks_dir for study command.")
        env_args = {
            "task": task_id,
            "tasks_dir": str(tasks_dir),
            "max_turns": 100,
            "timeout_seconds": 3600.0,
        }
        job_dir = run_dir / str(pair["pair_id"]) / condition
        output_dir = job_dir / "vf_outputs"
        command = [
            "uv",
            "run",
            "vf-eval",
            "general-agent-solver-local",
            "-p",
            "prime",
            "-m",
            model_id,
            "-n",
            "1",
            "-r",
            str(rollouts),
            "-s",
            "-d",
            "-v",
            "-o",
            str(output_dir),
            "-a",
            json.dumps(env_args, sort_keys=True),
        ]
        return command, job_dir

    def _select_prime_model(self) -> str:
        api_key = os.environ.get("PRIME_API_KEY")
        if not api_key:
            return self._select_prime_model_from_cli()
        request = urllib.request.Request(
            "https://api.pinference.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        team_id = os.environ.get("PRIME_TEAM_ID")
        if team_id:
            request.add_header("X-Prime-Team-ID", team_id)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise AdapterError(
                "prime_model_preflight_failed",
                "Prime Inference model listing failed.",
                {"status": exc.code, "body": detail},
            ) from exc
        except urllib.error.URLError as exc:
            raise AdapterError(
                "prime_model_preflight_failed",
                f"Prime Inference model listing failed: {exc.reason}",
            ) from exc
        data = json.loads(body.decode("utf-8"))
        rows = data.get("data", [])
        if not isinstance(rows, list):
            raise AdapterError("prime_model_preflight_failed", "Prime model list response did not contain data list.")
        selected = self._preferred_model_id(rows, ("kimi", "moonshot"))
        if selected is None:
            selected = self._preferred_model_id(rows, ("glm", "z.ai", "zai", "z-ai"))
        if selected is None:
            raise AdapterError(
                "prime_model_preflight_failed",
                "No Kimi/Moonshot or GLM/Z.ai model was found in the Prime Inference model list.",
            )
        return selected

    def _select_prime_model_from_cli(self) -> str:
        try:
            process = subprocess.run(
                ["prime", "inference", "models", "--output", "json", "--plain"],
                cwd=self.spec.root,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdapterError(
                "missing_credentials",
                "PRIME_API_KEY is missing and Prime CLI model listing was unavailable.",
            ) from exc
        if process.returncode != 0:
            raise AdapterError(
                "missing_credentials",
                "PRIME_API_KEY is missing and Prime CLI model listing failed.",
                {"stderr": process.stderr[-2000:]},
            )
        try:
            data = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError("prime_model_preflight_failed", "Prime CLI model listing did not return JSON.") from exc
        rows = data.get("data") or data.get("models") or []
        if not isinstance(rows, list):
            raise AdapterError("prime_model_preflight_failed", "Prime CLI model list response did not contain data list.")
        selected = self._preferred_model_id(rows, ("kimi", "moonshot"))
        if selected is None:
            selected = self._preferred_model_id(rows, ("glm", "z.ai", "zai", "z-ai"))
        if selected is None:
            raise AdapterError(
                "prime_model_preflight_failed",
                "No Kimi/Moonshot or GLM/Z.ai model was found in the Prime CLI model list.",
            )
        return selected

    def _preferred_model_id(self, rows: list[Any], needles: tuple[str, ...]) -> str | None:
        matches = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            model_id = row["id"]
            lower = model_id.casefold()
            if not any(needle in lower for needle in needles):
                continue
            created = row.get("created")
            matches.append((float(created) if isinstance(created, int | float) else 0.0, model_id))
        if not matches:
            return None
        return sorted(matches)[-1][1]

    def _record_selected_model(self, manifest_path: Path, model_id: str) -> None:
        manifest = self._load_manifest(manifest_path)
        model_selection = manifest.get("model_selection")
        if not isinstance(model_selection, dict):
            model_selection = {}
        model_selection.update(
            {
                "provider": "prime",
                "selected_model_id": model_id,
                "selected_at": datetime.now(UTC).isoformat(),
                "selection_status": "selected_by_prime_preflight",
            }
        )
        manifest["model_selection"] = model_selection
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _pass_rate_from_eval_text(self, text: str, metrics: dict[str, float]) -> float | None:
        for key in ("reward", "score", "verify", "accuracy", "mean_reward", "pass_rate"):
            value = metrics.get(key)
            if isinstance(value, int | float):
                return float(value)
        for key, value in metrics.items():
            lower = key.casefold()
            if lower.endswith("pass_rate") or lower.endswith("accuracy") or lower.endswith("reward"):
                return float(value)
        for pattern in (
            r"pass@1\s*=\s*(-?\d+(?:\.\d+)?)",
            r"pass[_ -]?rate[=: ]+(-?\d+(?:\.\d+)?)",
            r"mean[_ -]?reward[=: ]+(-?\d+(?:\.\d+)?)",
            r"reward[=: ]+(-?\d+(?:\.\d+)?)",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _merge_condition_results(
        self,
        pairs: list[dict[str, Any]],
        results: list[dict[str, Any]],
        validation_records: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        by_key = {(str(result.get("pair_id")), str(result.get("condition"))): result for result in results}
        validation_records = validation_records or {}
        merged = []
        for pair in pairs:
            pair_id = str(pair["pair_id"])
            base = by_key.get((pair_id, "base"))
            evolved = by_key.get((pair_id, "evolved"))
            validation_record = validation_records.get(pair_id, {})
            validation_metrics = validation_record.get("metrics", {}) if isinstance(validation_record, dict) else {}
            merged.append(
                {
                    "pair_id": pair_id,
                    "base_task_id": pair["base_task_id"],
                    "candidate_task_id": pair.get("candidate_task_id"),
                    "included": True,
                    "mutation_kind": pair.get("mutation_kind"),
                    "atomic_change": pair.get("atomic_change"),
                    "base_recorded_local_pass_rate": pair.get("base_recorded_local_pass_rate"),
                    "base": base or {},
                    "evolved": evolved or {},
                    "base_pass_rate": base.get("pass_rate") if isinstance(base, dict) else None,
                    "evolved_pass_rate": evolved.get("pass_rate") if isinstance(evolved, dict) else None,
                    "metrics": validation_metrics if isinstance(validation_metrics, dict) else {},
                }
            )
        return merged

    def _candidate_task_dir(self, p: dict[str, Any]) -> Path:
        raw = p.get("candidate_task_dir") or p.get("candidate_dir")
        if not isinstance(raw, str) or not raw.strip():
            raise AdapterError("invalid_input", "evaluate_candidate requires candidate_task_dir.")
        path = self.spec.resolve(raw)
        self._ensure_inside(self.spec.root, path, "candidate_task_dir")
        return path

    def _task_dir_by_id(self, task_id: str) -> Path:
        task_dir = self._tasks_dir(required=True) / task_id
        if not task_dir.exists():
            raise AdapterError("invalid_input", f"Unknown General Agent base task: {task_id}")
        return task_dir

    def _tasks_dir(self, *, required: bool) -> Path | None:
        source = self._source()
        raw = source.env_args.get("tasks_dir")
        if raw:
            path = Path(str(raw))
            resolved = path.resolve() if path.is_absolute() else self.spec.resolve(str(raw))
            if not resolved.exists():
                raise AdapterError("invalid_spec", f"Configured General Agent tasks_dir does not exist: {raw}")
            return resolved
        for root in self._search_roots():
            candidate = root / "environments" / "general_agent" / "tasks"
            if candidate.exists():
                return candidate.resolve()
        if required:
            raise AdapterError("invalid_spec", "Could not find environments/general_agent/tasks.")
        return None

    def _general_agent_package_root(self) -> Path | None:
        for root in self._search_roots():
            candidate = root / "environments" / "general_agent"
            if (candidate / "general_agent").exists():
                return candidate.resolve()
        return None

    def _search_roots(self) -> list[Path]:
        roots = [self.spec.root.resolve(), *self.spec.root.resolve().parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
        unique: list[Path] = []
        seen = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            unique.append(root)
        return unique

    def _load_attr(self, path: Path, attr: str) -> Any:
        package_root = self._general_agent_package_root()
        if package_root is not None and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        module_name = f"_research_env_general_agent_{path.parent.name}_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AdapterError("invalid_task", f"Could not load Python module: {self._relative_or_str(path)}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, attr, None)

    def _task_metadata(self, task_dir: Path) -> dict[str, Any]:
        with open(task_dir / "task.toml", "rb") as handle:
            raw = tomllib.load(handle)
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise AdapterError("invalid_task", f"task.toml metadata must be a mapping: {self._relative_or_str(task_dir)}")
        return metadata

    def _tool_names(self, task_dir: Path) -> list[str]:
        tools_cls = self._load_attr(task_dir / "tools.py", "TaskTools")
        db_cls = self._load_attr(task_dir / "tools.py", "TaskDB")
        if tools_cls is None or db_cls is None:
            return []
        tools = tools_cls(db_cls.load(task_dir / "db.json"))
        return sorted(tools.tool_methods)

    def _verify_score(self, verify_fn: Any, db: Any) -> float:
        try:
            return float(verify_fn(db))
        except Exception as exc:  # intentional: verifier failures are evidence
            raise AdapterError("invalid_task", f"verify(db) raised: {type(exc).__name__}: {exc}") from exc

    def _load_json(self, path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _file_sha256(self, path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        return sha256_bytes(path.read_bytes())

    def _ensure_inside(self, root: Path, path: Path, label: str) -> None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise AdapterError("forbidden", f"{label} must stay within the spec directory.") from exc

    def _relative_or_str(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return self._relative(path)
        except ValueError:
            return str(path)

    def _new_evolution_id(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"evo_{stamp}_{uuid4().hex[:8]}"
