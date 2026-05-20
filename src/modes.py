from __future__ import annotations

from typing import Any

from adapters.base import AdapterError
from envelope import params

EVIDENCE_MODES = ("evaluation", "analysis", "debug")


def evidence_mode(input_data: dict[str, Any]) -> str:
    p = params(input_data)
    raw = p.get("evidence_mode", p.get("mode"))
    if raw is None and p.get("include_reference") is True:
        return "analysis"
    mode = str(raw or "evaluation").strip().lower()
    if mode not in EVIDENCE_MODES:
        allowed = ", ".join(EVIDENCE_MODES)
        raise AdapterError("invalid_input", f"evidence_mode must be one of: {allowed}.")
    return mode


def references_visible(mode: str) -> bool:
    return mode in {"analysis", "debug"}


def debug_visible(mode: str) -> bool:
    return mode == "debug"
