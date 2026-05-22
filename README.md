# research-env

`research-env` is a CLI contract for turning an environment into a passive
research substrate for agents.

The environment keeps ownership of tools, tasks, verifiers, contracts, metrics,
and artifacts. The external agent keeps ownership of hypotheses, planning,
loops, and conclusions.

This repository includes the core CLI implementation and one General Agent
example with recorded evidence.

## CLI Contract

Every instrument follows the same boundary:

```text
stdin   JSON input object
stdout  one JSON envelope
stderr  logs/debug output
exit 0  valid envelope produced, including ok=false
```

Envelope fields:

```text
schema_version
ok
instrument
substrate_id
provenance
contract_status
call
result
metrics
evidence
artifacts
warnings
error
```

## Layout

```text
src/                         core CLI and adapters
general_agent/research_env.yaml
general_agent/compare_input.json
general_agent/inspect_input.json
general_agent/envelopes/zero_n30_study_envelope.json
general_agent/results/zero_n30_statistics.json
general_agent/results/atomic_non_degradation_hypothesis.json
```

## Install

```bash
uv pip install -e .
```

Or run directly from the repo:

```bash
uv run research-env --spec general_agent/research_env.yaml describe
```

## General Agent Example

The included General Agent example is a recorded `visible policy/tool` study.

Hypothesis:

```text
Making one implicit task rule explicit through a lookup tool should preserve or
improve general-agent-solver-local pass rate while preserving verifier validity.
```

Mutation shape:

```text
base task:
  use the original action directly

evolved task:
  add one action_policies relation
  add one list_action_policies lookup tool
  require policy_id on the original action
  add one gold lookup step before the action
```

Run shape:

```text
n = 30 base/evolved task pairs
solver = general-agent-solver-local
provider/model = Prime / moonshotai/kimi-k2.6
rollouts = 3 per condition
total eval jobs = 60
total rollouts = 180
```

Recorded result:

```text
n_pairs = 30
n_pass_rate_pairs = 30
base_mean_pass_rate = 0.6888666666666666
evolved_mean_pass_rate = 0.7666666666666667
mean_paired_pass_rate_drop = -0.07780000000000001
evolved_lt_base_count = 0
evolved_eq_base_count = 25
evolved_gt_base_count = 5
```

Interpretation:

```text
The evolved tasks did not make the solver worse. Making the implicit rule
explicit through a tool preserved or improved pass rate in this n=30 run.
```

Second checked hypothesis:

```text
For included General Agent visible-policy-affordance task pairs, the mutation is
structurally atomic and non-degrading: exactly one tool is added, exactly one
gold step is added, and evolved pass rate is never lower than paired base pass
rate.
```

Recorded result:

```text
n_pairs = 30
delta_tool_eq_1_count = 30
delta_gold_steps_eq_1_count = 30
non_degraded_pairs = 30
improved_pairs = 5
tied_pairs = 25
worsened_pairs = 0
```

## Command Loop

Describe the substrate contract:

```bash
uv run research-env --spec general_agent/research_env.yaml describe
```

Compare the recorded General Agent study envelope:

```bash
uv run research-env --spec general_agent/research_env.yaml compare-results < general_agent/compare_input.json
```

Inspect the saved second-hypothesis result artifact:

```bash
uv run research-env --spec general_agent/research_env.yaml inspect-artifact < general_agent/inspect_input.json
```

## Scope

This repository provides the CLI contract and one compact General Agent evidence
example. It does not include the full General Agent task corpus, candidate task
directories, native eval outputs, or Prime runtime assets needed to rerun the
live 60-job study from scratch.
