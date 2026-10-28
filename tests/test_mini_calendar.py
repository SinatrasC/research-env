"""Black-box tests: the research-env CLI against the mini-calendar verifiers v1 taskset.

Each test runs the installed `research-env` command on a fresh copy of examples/mini_calendar and
reads the envelope it prints. They need verifiers v1 and the example taskset installed:

    uv pip install -e . -e examples/mini_calendar/taskset && pytest
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verifiers.v1") is None or importlib.util.find_spec("mini_calendar") is None,
    reason="needs verifiers v1 and the mini-calendar example taskset installed",
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "mini_calendar"
BIN = Path(sys.executable).parent


@pytest.fixture()
def example(tmp_path: Path) -> Path:
    target = tmp_path / "mini_calendar"
    shutil.copytree(EXAMPLE, target, ignore=shutil.ignore_patterns("artifacts", "__pycache__", "*.egg-info"))
    return target


def call(example: Path, instrument: str, params: dict | None = None) -> dict:
    process = subprocess.run(
        [str(BIN / "research-env"), "--spec", str(example / "research_env.yaml"), instrument],
        input=json.dumps({"params": params or {}}),
        text=True,
        capture_output=True,
        timeout=900,
        env={**os.environ, "FAKE_MODEL_KEY": "unused"},
        check=False,
    )
    assert process.returncode == 0, process.stderr[-2000:]
    return json.loads(process.stdout)


def ok(envelope: dict) -> dict:
    assert envelope["ok"], envelope["error"]
    return envelope


def test_describe_probe_reports_the_tasksets_signals(example: Path) -> None:
    envelope = ok(call(example, "describe", {"probe": True}))
    probe = envelope["result"]["probe"]
    assert probe["task_count"] == 3
    assert probe["rewards"] == ["booked"]
    assert probe["has_gold_check"] is True
    assert envelope["contract_status"]["fixed_resources_unchanged"] is True


def test_evaluation_mode_hides_references_and_analysis_mode_shows_them(example: Path) -> None:
    hidden = ok(call(example, "sample-task", {"limit": 3}))["result"]["tasks"]
    assert [task["data"]["id"] for task in hidden] == ["t0", "t1", "t2"]
    for task in hidden:
        assert set(task["data"]) <= {"idx", "id", "name", "description", "prompt", "system_prompt"}
    shown = ok(call(example, "sample-task", {"limit": 1, "evidence_mode": "analysis"}))["result"]["tasks"][0]
    assert shown["data"]["gold"] == [["book_slot", {"person": "ana", "slot": "09:00"}]]


def test_evaluate_candidate_flags_bad_gold_and_presolved_tasks(example: Path) -> None:
    envelope = ok(
        call(
            example,
            "evaluate-candidate",
            {
                "base": {"taskset": {"config": {"variant": "base"}}},
                "candidate": {"taskset": {"config": {"variant": "evolved"}}},
            },
        )
    )
    base = {task["pair_key"]: task for task in envelope["result"]["base"]["tasks"]}
    assert base["t0"]["valid"] is True
    assert base["t1"]["error"] == "gold: invalid"
    assert base["t2"]["noop"] == "invalid" and base["t2"]["noop_reward"] == 1.0
    assert envelope["metrics"]["candidate_valid_rate"] == 1.0
    assert envelope["metrics"]["delta_valid_rate"] == pytest.approx(2 / 3)


def test_contract_status_detects_a_changed_taskset(example: Path) -> None:
    taskset = example / "taskset" / "mini_calendar" / "taskset.py"
    taskset.write_text(taskset.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    status = ok(call(example, "describe"))["contract_status"]
    assert status["fixed_resources_unchanged"] is False
    assert status["changed_fixed_resources"] == ["taskset:mini-calendar"]


def test_spec_without_an_explicit_runtime_is_refused(example: Path) -> None:
    spec = example / "research_env.yaml"
    spec.write_text(spec.read_text(encoding="utf-8").replace("type: subprocess", "type: somewhere"), encoding="utf-8")
    envelope = call(example, "describe")
    assert envelope["ok"] is False
    assert envelope["error"]["kind"] == "spec_error"
    assert "runtime.type" in envelope["error"]["message"]


@pytest.fixture()
def fake_model():
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8765)) == 0:
            pytest.skip("port 8765 is busy")
    server = subprocess.Popen(
        [sys.executable, str(EXAMPLE / "fake_model.py"), "--port", "8765"], stderr=subprocess.DEVNULL
    )
    for _ in range(50):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8765)) == 0:
                break
        time.sleep(0.1)
    yield
    server.terminate()
    server.wait(timeout=10)


def test_run_subject_and_compare_results_pair_rewards_by_task(example: Path, fake_model) -> None:
    evolved = {"taskset": {"config": {"variant": "evolved"}}}
    careful = ok(call(example, "run-subject", {"model": "careful", **evolved}))
    sloppy = ok(call(example, "run-subject", {"model": "sloppy", **evolved}))
    assert careful["metrics"]["mean_reward"] == 1.0
    assert sloppy["result"]["per_task"] == {"t0": 1.0, "t1": 0.0, "t2": 0.0, "t3": 1.0, "t4": 1.0}
    assert sloppy["metrics"]["errored_episodes"] == 0

    compared = ok(
        call(
            example,
            "compare-results",
            {"left_run": sloppy["result"]["run_dir"], "right_run": careful["result"]["run_dir"]},
        )
    )
    assert compared["result"]["kind"] == "eval"
    assert compared["metrics"]["n_pairs"] == 5
    assert compared["metrics"]["right_better_count"] == 2 and compared["metrics"]["tied_count"] == 3
    assert compared["metrics"]["mean_delta"] == pytest.approx(0.4)
