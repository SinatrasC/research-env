from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "research-env.v0"


@dataclass
class InstrumentPayload:
    result: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CallContext:
    instrument: str
    input_data: dict[str, Any]
    started: float = field(default_factory=time.monotonic)
    id: str = field(init=False)
    input_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = f"call_{self.instrument}_{uuid4().hex[:12]}"
        self.input_sha256 = sha256_json(self.input_data)

    @property
    def duration_seconds(self) -> float:
        return time.monotonic() - self.started


def params(input_data: dict[str, Any]) -> dict[str, Any]:
    value = input_data.get("params", input_data)
    return value if isinstance(value, dict) else {}


def success_envelope(
    *,
    substrate_id: str,
    context: CallContext,
    payload: InstrumentPayload,
    provenance: dict[str, Any] | None = None,
    contract_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "instrument": context.instrument,
        "substrate_id": substrate_id,
        "provenance": provenance or {},
        "contract_status": contract_status,
        "call": call_payload(context),
        "result": payload.result,
        "metrics": payload.metrics,
        "evidence": payload.evidence,
        "artifacts": payload.artifacts,
        "warnings": payload.warnings,
        "error": None,
    }


def error_envelope(
    *,
    instrument: str,
    substrate_id: str | None,
    context: CallContext | None,
    kind: str,
    message: str,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    contract_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "instrument": instrument,
        "substrate_id": substrate_id,
        "provenance": provenance or {},
        "contract_status": contract_status,
        "call": call_payload(context) if context else None,
        "result": {},
        "metrics": {},
        "evidence": [],
        "artifacts": [],
        "warnings": warnings or [],
        "error": {
            "kind": kind,
            "message": message,
            "details": details or {},
        },
    }


def call_payload(context: CallContext) -> dict[str, Any]:
    return {
        "id": context.id,
        "input_sha256": context.input_sha256,
        "duration_seconds": round(context.duration_seconds, 6),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(text)
