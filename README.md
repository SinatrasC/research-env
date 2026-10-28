# research-env

`research-env` is a CLI contract for turning an environment into a passive
research substrate for agents.

The environment keeps ownership of tools, tasks, verifiers, contracts, metrics,
and artifacts. The external agent keeps ownership of hypotheses, planning,
loops, and conclusions.

Version 0.2 targets [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1
(`import verifiers.v1 as vf`): the substrate is a v1 **taskset**, and the
instruments are thin, audited wrappers around verifiers' own tools
(`vf-validate`, `vf-eval`), read back from the run directories verifiers writes.
The v0 implementation (pre-v1 verifiers and the General Agent study) is tagged `v0`.

## CLI contract

Every instrument follows the same boundary:

```text
stdin   JSON input object, {"params": {...}}
stdout  one JSON envelope
stderr  logs/debug output
exit 0  valid envelope produced, including ok=false
```

Envelope fields: `schema_version` (`research-env.v1`), `ok`, `instrument`,
`substrate_id`, `provenance`, `contract_status`, `call`, `result`, `metrics`,
`evidence`, `artifacts`, `warnings`, `error`.

## Instruments on a verifiers v1 taskset

| Instrument | What it does |
|---|---|
| `describe` | The contract, substrate and fixed evaluation shape. `probe: true` also loads the taskset and reports its task count, reward and metric names, toolsets and whether tasks define a gold check. |
| `sample-task` | Tasks from the taskset. In `evaluation` mode (the default) only `idx`, `id`, `name`, `description`, `prompt` and `system_prompt` are shown; answers, gold actions and seeded state appear only in `analysis` or `debug` mode. |
| `evaluate-candidate` | Runs `vf-validate` (setup, gold and noop checks) on a candidate taskset, optionally against a `base` one, and reports valid rates and per-task verdicts. A task is invalid if its reference solution fails (`gold`) or if it already scores 1 with no agent action (`noop`). |
| `run-subject` | Runs `vf-eval` for one subject model under the evaluation shape the spec fixes (harness, runtime, sampling, rollouts, turns). Reads `traces.jsonl`: per-episode reward (score x weight, as `Trace.reward`), per-task means, tokens and agent time. `dry_run: true` only resolves the config. |
| `inspect-artifact` | Reads a file under the artifact root, with its sha256. |
| `compare-results` | Pairs two `run-subject` runs (rewards) or two `evaluate-candidate` runs (validity) task by task, by `TaskData.id`, and reports the paired mean, a bootstrap 95% interval and an exact sign test. |

`run-subject` and `evaluate-candidate` accept `taskset: {id, config}` to run a
candidate taskset in place of the spec's (for example an evolved variant).

## Spec (`research_env.yaml`)

```yaml
schema_version: research-env.v1
id: mini_calendar_booking
source_env: mini-calendar
adapter:
  module: adapters.verifiers_v1
contracts:
  fixed:
    - id: taskset:mini-calendar
      path: taskset/mini_calendar     # hashed; contract_status flags any change
      sha256: <recorded baseline>
    - harness:null
  mutable: [artifacts/**]
  substrate_mutation: false
execution:
  artifact_root: artifacts
substrate:
  taskset: {id: mini-calendar, config: {variant: base}}
  runtime: {type: subprocess}         # required: subprocess | docker | prime | custom
  eval:
    harness: {id: "null"}
    sampling: {temperature: 0.0}
    num_rollouts: 1
    max_turns: 6
    client: {base_url: http://127.0.0.1:8765/v1, api_key_var: FAKE_MODEL_KEY}
```

The runtime has no default because `prime` starts billed remote sandboxes.
Fixed resources with a `path` are hashed on every call (bytecode and build
metadata excluded); `provenance.substrate` also records the verifiers and
taskset package versions and hashes of the taskset config and evaluation shape.

## Example: mini-calendar

`examples/mini_calendar/` is the General Agent task shape (a small database,
stateful MCP tools, a gold action sequence, `verify(db)` as the reward) written
as a v1 taskset with two variants:

- `base`: three tasks. `t0` is valid, `t1`'s gold books the wrong slot, `t2` is
  already solved before any action.
- `evolved`: five valid tasks; in two of them the requested slot is not the first
  free one.

`fake_model.py` is a scripted OpenAI-compatible endpoint, so `run-subject`
produces real traces locally: `careful` books the requested slot, `sloppy` books
the first free one. Both call the task's real tools through verifiers'
interception server.

```bash
uv pip install -e . -e examples/mini_calendar/taskset
bash examples/mini_calendar/demo.sh
```

Recorded output (2026-10-28, verifiers 0.2.0.dev0 at upstream `60290d436`, about 50 s; values rounded):

```text
describe               {'task_count': 3}
sample_evaluation      {'returned': 3}
evaluate_evolved       {'base_valid_rate': 0.333, 'candidate_valid_rate': 1.0, 'delta_valid_rate': 0.667}
run_careful            {'episodes': 5, 'errored_episodes': 0, 'mean_reward': 1.0}
run_sloppy             {'episodes': 5, 'errored_episodes': 0, 'mean_reward': 0.6}
compare_subjects       {'mean_delta': 0.4, 'n_pairs': 5, 'right_better_count': 2, 'sign_test_two_sided_p': 0.5, 'tied_count': 3}
```

The base validation rejects `t1` (`gold: invalid`) and `t2` (`noop: passes
untouched (reward=1)`). The comparison is honest about its size: careful beats
sloppy on the 2 tasks built to separate them, and two non-ties cannot give a
sign-test p below 0.5.

## Tests

```bash
uv pip install -e . -e examples/mini_calendar/taskset pytest
pytest
```

Black-box: each test runs the `research-env` command on a fresh copy of the
example and checks the envelope. They skip when verifiers v1 or the example
taskset is not installed.

## Changes from v0

- The verifiers adapter is rewritten for v1. v0 called `verifiers.load_environment`,
  read `dataset` rows by `question`/`answer` keys and parsed `vf-eval`'s printed
  output; none of these exist in v1.
- The General Agent adapter and its recorded n=30 study are removed (tag `v0`):
  v1's `vf-validate` now does the gold-replay and untouched-state checks the
  adapter implemented, and the study's corpus was never in this repository.
- The spec schema is `research-env.v1`: `substrate.taskset`, `substrate.runtime`
  and `substrate.eval` replace `substrate.task_source`.
- Paired statistics moved to `src/stats.py`, unchanged (same bootstrap seed).

## Limits

- Not yet run against a remote runtime, a container harness or a hub taskset;
  the example uses the local `subprocess` runtime and the `null` harness.
- No compute or energy metering in the envelope yet.
- verifiers v1 is moving quickly; pin a verifiers commit as a fixed resource for
  long-running studies.
