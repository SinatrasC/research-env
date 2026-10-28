from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from contextlib import redirect_stdout
from typing import Any, TextIO

from adapters.base import Adapter, AdapterError
from envelope import CallContext, error_envelope, success_envelope
from provenance import build_contract_status, build_provenance
from spec import SpecError, load_spec

__version__ = "0.2.0"

INSTRUMENTS = (
    "describe",
    "sample-task",
    "evaluate-candidate",
    "run-subject",
    "inspect-artifact",
    "compare-results",
)


class CliUsageError(ValueError):
    pass


class EnvelopeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def main(argv: list[str] | None = None) -> int:
    envelope_stdout = sys.stdout
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
    except CliUsageError as exc:
        context = CallContext(instrument="unknown", input_data={})
        with redirect_stdout(sys.stderr):
            provenance, contract_status, warnings = _safe_metadata(None)
        _write_json(
            error_envelope(
                instrument="unknown",
                substrate_id=None,
                context=context,
                kind="cli_usage_error",
                message=str(exc),
                warnings=warnings,
                provenance=provenance,
                contract_status=contract_status,
            ),
            out=envelope_stdout,
        )
        return 2

    instrument = args.instrument.replace("-", "_")
    raw_input = _read_stdin()

    try:
        input_data = _parse_input(raw_input)
    except ValueError as exc:
        context = CallContext(instrument=instrument, input_data={"raw_stdin": raw_input})
        with redirect_stdout(sys.stderr):
            provenance, contract_status, warnings = _safe_metadata(None)
        _write_json(
            error_envelope(
                instrument=instrument,
                substrate_id=None,
                context=context,
                kind="invalid_json",
                message=str(exc),
                warnings=warnings,
                provenance=provenance,
                contract_status=contract_status,
            ),
            out=envelope_stdout,
        )
        return 0

    context = CallContext(instrument=instrument, input_data=input_data)
    spec = None
    try:
        with redirect_stdout(sys.stderr):
            spec = load_spec(args.spec)
            adapter = load_adapter(spec.adapter.module, spec)
            method = getattr(adapter, instrument)
            payload = method(input_data)
            provenance, contract_status, warnings = _safe_metadata(spec)
        payload.warnings.extend(warnings)
        _write_json(
            success_envelope(
                substrate_id=spec.id,
                context=context,
                payload=payload,
                provenance=provenance,
                contract_status=contract_status,
            ),
            out=envelope_stdout,
        )
        return 0
    except (SpecError, AdapterError) as exc:
        kind = exc.kind if isinstance(exc, AdapterError) else "spec_error"
        details = exc.details if isinstance(exc, AdapterError) else {}
        with redirect_stdout(sys.stderr):
            provenance, contract_status, warnings = _safe_metadata(spec)
        _write_json(
            error_envelope(
                instrument=instrument,
                substrate_id=spec.id if spec else None,
                context=context,
                kind=kind,
                message=str(exc),
                details=details,
                warnings=warnings,
                provenance=provenance,
                contract_status=contract_status,
            ),
            out=envelope_stdout,
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI envelope boundary
        with redirect_stdout(sys.stderr):
            traceback.print_exc(file=sys.stderr)
            provenance, contract_status, warnings = _safe_metadata(spec)
        _write_json(
            error_envelope(
                instrument=instrument,
                substrate_id=spec.id if spec else None,
                context=context,
                kind="internal_error",
                message=str(exc),
                warnings=warnings,
                provenance=provenance,
                contract_status=contract_status,
            ),
            out=envelope_stdout,
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = EnvelopeArgumentParser(prog="research-env")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--spec", help="Path to research_env.yaml. Defaults to upward discovery from cwd.")
    parser.add_argument("instrument", choices=INSTRUMENTS)
    return parser


def load_adapter(module_name: str, spec) -> Adapter:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise AdapterError(
            "adapter_import_error",
            f"Could not import adapter module or dependency: {exc.name}",
            {"module": module_name, "missing": exc.name},
        ) from exc
    except ImportError as exc:
        raise AdapterError(
            "adapter_import_error",
            f"Could not import adapter module: {module_name}",
            {"module": module_name, "error": str(exc)},
        ) from exc

    try:
        adapter_cls = getattr(module, "Adapter")
    except AttributeError as exc:
        raise AdapterError(
            "invalid_adapter",
            f"{module_name} does not define Adapter.",
            {"module": module_name},
        ) from exc

    try:
        adapter = adapter_cls(spec)
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(
            "adapter_initialization_error",
            f"Could not initialize adapter {module_name}.Adapter: {exc}",
            {"module": module_name, "error": str(exc)},
        ) from exc

    if not isinstance(adapter, Adapter):
        raise AdapterError(
            "invalid_adapter",
            f"{module_name}.Adapter must inherit adapters.base.Adapter.",
            {"module": module_name},
        )
    return adapter


def _read_stdin() -> str:
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def _parse_input(raw_input: str) -> dict[str, Any]:
    text = raw_input.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object.")
    return value


def _safe_metadata(spec) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    warnings: list[str] = []

    try:
        provenance = build_provenance(spec)
    except Exception as exc:
        provenance = {}
        warnings.append(f"provenance collection failed: {exc}")

    try:
        contract_status = build_contract_status(spec)
    except Exception as exc:
        contract_status = None
        warnings.append(f"contract status collection failed: {exc}")

    return provenance, contract_status, warnings


def _write_json(data: dict[str, Any], *, out: TextIO | None = None) -> None:
    stream = out or sys.stdout
    json.dump(data, stream, indent=2, sort_keys=True)
    stream.write("\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
