from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from envelope import InstrumentPayload
from spec import ResearchEnvSpec


class AdapterError(ValueError):
    def __init__(self, kind: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.details = details or {}


@dataclass
class Adapter:
    spec: ResearchEnvSpec

    def describe(self, input_data: dict[str, Any]) -> InstrumentPayload:
        raise NotImplementedError

    def sample_task(self, input_data: dict[str, Any]) -> InstrumentPayload:
        raise NotImplementedError

    def evaluate_candidate(self, input_data: dict[str, Any]) -> InstrumentPayload:
        raise NotImplementedError

    def run_subject(self, input_data: dict[str, Any]) -> InstrumentPayload:
        raise NotImplementedError

    def inspect_artifact(self, input_data: dict[str, Any]) -> InstrumentPayload:
        raise NotImplementedError

    def compare_results(self, input_data: dict[str, Any]) -> InstrumentPayload:
        raise NotImplementedError
