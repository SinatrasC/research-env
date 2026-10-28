#!/usr/bin/env bash
# The mini-calendar demo: every research-env instrument against a verifiers v1 taskset, locally.
#
#   uv pip install -e . -e examples/mini_calendar/taskset    # research-env + the example taskset
#   bash examples/mini_calendar/demo.sh
#
# No model endpoint or sandbox is billed: tasks run in the `subprocess` runtime and the subject is
# the scripted fake model in fake_model.py. Envelopes are written to examples/mini_calendar/artifacts/demo/.
set -euo pipefail
cd "$(dirname "$0")"
export FAKE_MODEL_KEY=unused
OUT=artifacts/demo
mkdir -p "$OUT"

python fake_model.py --port 8765 2> "$OUT/fake_model.log" &
FAKE=$!
trap 'kill $FAKE 2>/dev/null || true' EXIT

call() {  # call NAME INSTRUMENT INPUT_JSON
  echo "$3" | research-env --spec research_env.yaml "$2" > "$OUT/$1.json"
  python - "$OUT/$1.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
name = sys.argv[1].rsplit("/", 1)[-1][:-5]
if not d["ok"]:
    sys.exit(f"{name}: {d['error']['kind']}: {d['error']['message']}")
keep = ("task_count", "returned", "candidate_valid_rate", "base_valid_rate", "delta_valid_rate", "mean_reward",
        "episodes", "errored_episodes", "n_pairs", "mean_delta", "right_better_count", "tied_count",
        "sign_test_two_sided_p")
print(f"{name:22s}", {k: v for k, v in d["metrics"].items() if k in keep})
PY
}
run_dir() { python -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["run_dir"])' "$OUT/$1.json"; }

call describe describe '{"params": {"probe": true}}'
call sample_evaluation sample-task '{"params": {"limit": 3}}'
call evaluate_evolved evaluate-candidate \
  '{"params": {"base": {"taskset": {"config": {"variant": "base"}}}, "candidate": {"taskset": {"config": {"variant": "evolved"}}}}}'
call run_careful run-subject '{"params": {"model": "careful", "taskset": {"config": {"variant": "evolved"}}}}'
call run_sloppy run-subject '{"params": {"model": "sloppy", "taskset": {"config": {"variant": "evolved"}}}}'
call compare_subjects compare-results \
  "{\"params\": {\"left_run\": \"$(run_dir run_sloppy)\", \"right_run\": \"$(run_dir run_careful)\"}}"
echo "envelopes: $(pwd)/$OUT"
