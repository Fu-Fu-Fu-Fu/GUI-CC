# Offline Reference-Action Track

Autoregressive evaluation of GUI world models over 500 mobile GUI trajectories and 4,905
transitions (Table 2 in the paper).

## Dataset

The offline evaluation set contains 500 trajectories and 4,905 transitions, derived from
GUI-Odyssey. Download it from [🤗 minuzero/GUI-CC](https://huggingface.co/datasets/minuzero/GUI-CC)
into `data/`.

### Layout

```text
data/
├── offline_samples.jsonl     # 500 lines, one sample per line; line order = sample order
└── offline_data/<id>/        # id ranges over 001-500
    ├── reference_trajectory.jsonl
    ├── initial.png           # frame 0
    └── step_XXX_after.png    # the screen after step XXX
```

The *before* frame of a step is not stored separately: step 0's before frame is `initial.png`, and
every later step's before frame is the previous step's after frame. The two are byte-identical
upstream, and the evaluation code derives them that way.

### `offline_samples.jsonl` fields

| Field | Meaning |
|---|---|
| `sample_id` | task directory name (`001`-`500`) |
| `task_instruction` | task instruction; the input to both rollout and evaluation |
| `task_meta` | one-line summary of the task |
| `n_steps` | number of steps, equal to the number of reference trajectory lines |
| `source_dataset` / `source_episode_id` | upstream GUI-Odyssey dataset and episode id |
| `source_category` / `source_apps` | upstream task category and the apps involved |

### `reference_trajectory.jsonl` fields (one line per step)

| Field | Meaning |
|---|---|
| `step_id` | consecutive from 0; matches the screenshot `step_{step_id:03d}_after.png` |
| `semantic_action` | the semantic action |

Which fields a `semantic_action` carries depends on its type:

| Field | Meaning | Present for |
|---|---|---|
| `type` | `tap` / `scroll` / `long_press` / `navigate_home` / `navigate_back` | all |
| `target` | text description of the target element | all |
| `low_level_instruction` | low-level instruction for this step | tap / scroll / long_press |
| `source_coord` / `source_coord_norm` | tap point, in pixels and in per-mille | tap / long_press |
| `grounding.bbox` | bounding box of the target element | tap / long_press |
| `direction` / `start_norm` / `end_norm` | swipe direction and endpoints | scroll |

Capture environment: Android 14 (SDK 34) emulator at four device resolutions,
720x1280 / 1080x2400 / 1344x2992 / 1440x3120 (176 / 124 / 109 / 91 samples).

### Validation

```bash
python scripts/validate_data.py
```

Checks that the jsonl and the directories agree, that step counts match, that `step_id` is
consecutive, and that every image exists and decodes (online data included).

## Evaluation

### 1. Rollout protocol

The world model predicts the next GUI screenshot for each retained semantic action. Step 0 takes
`initial.png` as input; every later step takes the model's own previous prediction. The reference
action sequence is fixed throughout. Two history settings are supported:

- `WM-NoHist`: the harness passes only the current screen and the current action.
- `WM-FullHist`: the harness additionally passes the configured window of recent interaction
  history.

These settings describe what the harness supplies, not a property of the model. Every world model
evaluated in the paper consumes one step at a time, so `WM-FullHist` is a harness-side control that
supplies context those models cannot carry themselves. A model that maintains its own cross-step
state should be run under `WM-NoHist`, keeping that state inside its adapter instance; the harness
calls it once per step in trajectory order.

HTML/code world models and image/diffusion world models share the same sample set and evaluator but
differ in inference adapter and output artifact. The implementation lives in `offline/` and
`utils/adapters/`; the prompts live in `utils/prompts/`.

```bash
python -m offline.rollout --model code2world --setting WM-NoHist
python -m offline.cli     --model code2world --setting WM-NoHist
EVALUATE_AFTER_RUN=1 bash scripts/run_all_offline.sh   # every configured row
```

### 2. Metrics

The offline evaluator reports ten normalized metrics:

| Metric | Meaning | Granularity |
|---|---|---|
| `S_ele` | element alignment against the reference next state | transition |
| `S_lay` | layout integrity against the reference next state | transition |
| `S_sig` | SigLIP similarity | transition |
| `S_dino` | DINOv2 similarity | transition |
| `S_ad` | action adherence | transition |
| `S_id` | action identifiability | transition |
| `S_use` | GUI usability and rendering validity | transition |
| `S_cp` | cross-step context persistence | trajectory |
| `S_rd` | action-controlled rollout dynamics | trajectory |
| `S_rap` | ordered reference-action support and terminal completion | trajectory |

Per trajectory, the transition metrics are averaged first, then the three trajectory metrics are
computed. `Overall` is the unweighted mean of the ten metrics.

Scales and normalization: `S_ele` and `S_lay` are 1-10 ratings mapped by `(x-1)/9`; `S_ad` is a
0-10 rating divided by 10; `S_id` is a binary action-category match; `S_use`, `S_cp` and `S_rd` are
each the sum of five binary criteria divided by 5; `S_rap` is an ordered-prefix ratio; `S_sig` and
`S_dino` are encoder cosines clamped to `[0, 1]`.

The judge model comes from `judge_model` in `utils/configs/offline.json` and can be overridden with
`--judge-model`. `S_sig` and `S_dino` are computed locally by SigLIP and DINOv2 (configured through
the `VISUAL_SIM_*` environment variables). To compute only those two, without any API call:

```bash
python scripts/run_offline_local_metrics.py --model code2world --setting WM-NoHist
```

### 3. Failure attribution and aggregation

- A full evaluation (no `--sample-ids`) fixes the denominator at 500 episodes.
- **Model failure** (parse failure, empty output, `failure_class: "model"`): every metric for that
  episode scores 0, the episode stays in the fixed denominator, and aggregation is not blocked.
- **Infrastructure failure** (timeout, OOM, unreachable service, and so on): blocks official
  aggregation; fix and rerun.
- Passing `--sample-ids 001,002` or `--subset N` evaluates a subset; results go to separate
  `_*_partial.json` files and no official `Overall` is produced. `--subset N` takes a fixed, evenly
  spaced subset that is identical across models (see `utils/subset.py`).

### 4. Score representation

Per-sample `metrics` lie in `[0, 1]`. `paper_scores` multiplies by 100 and keeps one decimal;
intermediate results are never rounded early. `scripts/collect_results.py` aggregates every
configured row into `outputs/results/offline.{json,csv}`.

### 5. Output layout

```text
outputs/offline/
├── predictions/<model>/<nohist|fullhist>/
│   ├── run.json              # model configuration and commit for this rollout
│   ├── summary.json          # per-sample completion and failure records
│   └── <sample_id>/step_*/pred.png ...
└── evaluation/<model>/<nohist|fullhist>/
    ├── _preflight.json
    ├── _results.json
    ├── _aggregate.json
    └── <sample_id>/evaluation.json
```

One output directory accepts one rollout configuration only (`run.json`'s `run_sha256` must match);
use a new `--output-root` for a different configuration. Evaluation results are cached per episode
in `evaluation.json`; a change in configuration or prompt invalidates the cache, and `--force`
re-evaluates unconditionally.

### HTML rendering

The Code2World renderer shipped here is a strict port of the implementation released with the model.
Its input is the model's HTML output and its output is the PNG fed into the next step. The port is
kept as-is, including the following known behavior.

The renderer's reference size is always the sample's native resolution (the size of
`data/offline_data/<id>/initial.png`), identical at every step. Rollout therefore resizes each
predicted frame back to the native resolution before feeding it into the next step
(`resized_to_native` is recorded on the corresponding step in `summary.json`). The online track does
the same.

Transient browser faults (screenshot protocol errors and similar) reopen the browser and retry up to
three times; a persistent failure is recorded as an infrastructure failure and blocks aggregation
rather than being charged to the world model. A page load timeout is *not* infrastructure: it means
the model wrote HTML referencing resources that cannot load, which counts as a model failure scoring
0.

## Multi-GPU sharding

To generate one model/setting in parallel across GPUs, run the full rollout in a separate output
directory per shard and merge deterministically at the end. Sharding is round-robin over sample
order (`partition_sample_ids` in `offline/sharding.py`), so a shard's sample set is stable
regardless of shard count.

### Running shards

One process per shard (four shards, one GPU each):

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m offline.rollout \
    --model qwen_image_edit --setting WM-NoHist \
    --shard-count 4 --shard-index $i &
done
wait
```

Shard output goes to `outputs/offline/predictions/.shards/<model>-<hist>/shard-XXXXX-of-YYYYY/`,
isolated from the others. Rerunning the same command after an interruption skips finished samples.

### Merging

```bash
python -m offline.sharding merge \
  --model qwen_image_edit --setting WM-NoHist --shard-count 4
```

The merge verifies that every shard's run configuration agrees, that each sample was produced by
exactly one shard, and that its `pred.png` files are complete. It then atomically publishes the
sample directories together with `run.json` and `summary.json` into
`outputs/offline/predictions/<model>/<nohist|fullhist>/`, refusing to overwrite an existing target.
Only the merged directory is handed to the evaluator.
