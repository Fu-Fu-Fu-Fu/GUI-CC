# Online Agent-Loop Track

Closed-loop evaluation of a GUI agent over 200 tasks (Table 3 in the paper).

## Dataset

The online benchmark contains 200 tasks, each paired with exactly one initial-state screenshot.
Download it from [🤗 minuzero/GUI-CC](https://huggingface.co/datasets/minuzero/GUI-CC) into `data/`.

### Layout

```text
data/
├── online_samples.jsonl      # 200 lines, one task per line; line order = task order
└── online_data/<id>/         # id ranges over 001-200
    └── initial.png           # frame 0 for that task
```

### `online_samples.jsonl` fields (one task per line)

| Field | Meaning |
|---|---|
| `task_id` | task directory name (`001`-`200`) |
| `source_task_name` | descriptive name used during collection (e.g. `gmail_dark_theme_then_relaunch`) |
| `template_id` | task template (18 in total) |
| `category` | task-level label (6 app-domain categories plus `Cross-App`) |
| `apps` | applications involved |
| `step_budget` | maximum number of agent decisions (a terminal action counts as one) |
| `instruction` | task instruction |
| `initial_state` | text description of the initial state |
| `milestones` | ordered milestone list, the basis for `S_mp`; each entry is `{id, type, assertion}`, `type` is always a list of strings drawn from progress / terminal_success / memory_persistence, and the last milestone always contains terminal_success |

`initial.png` is a concrete frame-0 GUI screenshot, **not** a random seed. The runner feeds it to the
agent as the first observation; every later observation comes from a world-model prediction. A task
definition and its initial screenshot together form one evaluation case; neither alone is a complete
sample.

### Statistics

| Statistic | Value |
|---|---|
| Tasks / initial screenshots | 200 / 200, strictly one-to-one, all 1080x2400 |
| Average step budget | 15.74 (min 6, max 24) |
| Apps | 30 |
| App categories | 6 |
| Templates | 18 |
| Average milestones | 3.97 (min 2, max 5) |

15 cases carry the `category` value `Cross-App`. That is a task-level label for cross-app workflows,
not a seventh app domain; the six app domains are Communication, Finance & Business, Media &
Entertainment, Shopping & Delivery, System & Utility, and Travel & Navigation. The exact task-label
distribution is Communication 27, Finance & Business 19, Media & Entertainment 41, Shopping &
Delivery 24, System & Utility 41, Travel & Navigation 33, Cross-App 15.

### Validation

```bash
python scripts/validate_data.py
```

## Evaluation

### 1. Closed-loop protocol

The online agent is single-stage: at each step the planner looks at the current screenshot and
directly emits an action type and absolute pixel coordinates, following the OpenAI computer-use
convention. The world model then generates the next screen from that action. The loop stops at a
terminal action or when the task's step budget is exhausted, and progress is scored against the
ordered milestones in the task definition.

```bash
python -m online.rollout --model code2world --setting WM-Markov
python -m online.cli --rollout-dir outputs/online/code2world/markov
EVALUATE_AFTER_RUN=1 bash scripts/run_all_online.sh    # every configured row
```

Budget semantics: the budget caps the number of agent decisions, and a terminate action counts as
one. A rollout that exhausts the budget without terminating is still complete (N actions, N+1
frames). The judge frame cap `MAX_TRAJECTORY_FRAMES = 25` is aligned exactly with the maximum budget
of 24: every frame reaches the judge, and no subsampling occurs.

### 2. Metrics

| Metric | Meaning | Granularity |
|---|---|---|
| `S_ad` | action adherence | transition |
| `S_id` | action identifiability | transition |
| `S_use` | GUI usability and rendering validity | transition |
| `S_cp` | cross-step context persistence | trajectory |
| `S_rd` | action-controlled rollout dynamics | trajectory |
| `S_mp` | ordered milestone progress | trajectory |

`Overall` is the unweighted mean of the six metrics. `S_mp` is an ordered-prefix ratio: scoring stops
at the first milestone that fails, later milestones do not count, and the denominator is always the
total milestone count. The remaining scales match the offline track (`S_ad` 0-10 divided by 10,
`S_id` binary, `S_use` / `S_cp` / `S_rd` each the sum of five binary criteria divided by 5). The
judge model comes from `judge_model` in `utils/configs/online.json` and can be overridden with
`--judge-model`.

### 3. Failure attribution and aggregation

- Official aggregation fixes the denominator at 200 tasks.
- **Model failure** (`failure_class: "model"`): every metric for that task scores 0 and the task
  stays in the fixed denominator.
- **Infrastructure failure**: blocks official aggregation; fix and rerun.
- **Agent-side failures count as infrastructure**, not as model failures. The agent (planner) is a
  fixed component shared by every world model under test. Its own failures (API errors, exhausting
  retries without calling the `computer` tool, a gateway returning a different model) have nothing to
  do with the world model; recording them as model failures would let a broken agent produce a table
  that looks complete.
- Passing `--task-ids t1,t2` or `--subset N` evaluates a subset; results go to a separate
  `evaluation_partial.json` and no official result is produced. `--subset N` takes a fixed, evenly
  spaced subset that is identical across models (see `utils/subset.py`).

### 4. Score representation

Per-task `metrics` lie in `[0, 1]`. `paper_scores` multiplies by 100 and keeps one decimal;
intermediate results are never rounded early. `scripts/collect_results.py` aggregates every
configured row into `outputs/results/online.{json,csv}`.

### 5. Output layout

```text
outputs/online/<model>/<markov|fullhist>/
├── rollout_results.json          # _RUN configuration plus per-task records
├── <task_id>/
│   ├── step_<000..>/             # frame.png, action.json, agent_response.txt, agent_evidence.json
│   └── final_frame.png           # the last frame when the agent did not terminate
├── judge_cache/
└── evaluation_results.json
```

Evaluation results are cached per task in `judge_cache` and reused when the signature matches;
`--force` re-evaluates unconditionally.

## Multi-GPU sharding

To run one model/setting in parallel across GPUs, run the full rollout in a separate output
directory per shard and merge deterministically at the end. Sharding is round-robin over task order
(the line order of `online_samples.jsonl`).

### Running shards

```bash
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python -m online.rollout \
    --model code2world --setting WM-Markov \
    --output-root outputs/online \
    --shard-count 2 --shard-index $i &
done
wait
```

Shard output goes to `outputs/online/.shards/<model>-<hist>/shard-XXXXX-of-YYYYY/`. Rerunning the
same command after an interruption skips finished tasks.

### Merging

```bash
python -m online.sharding merge \
  --model code2world --setting WM-Markov \
  --output-root outputs/online --shard-count 2
```

The merge verifies that every shard's run configuration agrees and that each task was produced by
exactly one shard with complete artifacts (the per-step `step_<000..>/` directories and
`final_frame.png`). It then atomically publishes the task directories and the merged
`rollout_results.json` into `outputs/online/<model>/<markov|fullhist>/`, refusing to overwrite an
existing target. Only the merged directory is handed to the evaluator.
