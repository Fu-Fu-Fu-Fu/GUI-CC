# Repository Map

Every executable file in this release, and the minimal set each track depends on. The Markdown
templates under `utils/prompts/model/` and `utils/prompts/judge/` are described as groups; the
per-file mapping to the paper's figures is in `utils/prompts/README.md`.

## Root

| File | Purpose |
|---|---|
| `README.md` | project overview, installation, dataset, quick start, metrics, and citation |
| `LICENSE` | Apache License 2.0 (code); the dataset is CC BY 4.0 and lives on Hugging Face |
| `requirements.txt` | the single Python dependency list |
| `pytest.ini` | restricts pytest collection to `tests/` |
| `.gitignore` | excludes machine-local config, the dataset, outputs, caches, and bytecode |

## `utils/` — shared layer

| File | Purpose |
|---|---|
| `utils/config.py` | loads `utils/configs/paths.env` and reads JSON configuration |
| `utils/failure.py` | failure attribution: model failure versus infrastructure failure |
| `utils/io.py` | `load_json` and atomic JSON writes (the only implementation in the repo) |
| `utils/subset.py` | the fixed, evenly spaced `--subset N` sample subset, identical across models |
| `utils/configs/offline.json` | judge model, history window, and the offline model rows |
| `utils/configs/online.json` | planner and judge models, and the online model rows |
| `utils/configs/paths.env.example` | template for machine-local paths, endpoints, checkpoints, and revisions; copy to `paths.env`, which is not tracked |

### `utils/adapters/` — world-model interface

| File | Purpose |
|---|---|
| `registry.py` | maps a config model id to an adapter and resolves the inference configuration |
| `base.py` | shared request, parse, render, cache, and failure-evidence flow for HTML models |
| `diffusion_base.py` | shared request identity, generation, resize, and cache for direct image models |
| `annotate_action.py` | draws tap / long-press / scroll cues from the dataset coordinates |
| `html_to_png.py` | the Code2World Playwright rendering protocol |
| `closedmodel_adapter.py` | calls any OpenAI-compatible HTML-generating model with the paper's prompt |
| `code2world_adapter.py` | reference HTML adapter: Code2World messages, history prompt, and HTML parsing |
| `qwen_image_edit_adapter.py` | reference image adapter: Qwen-Image-Edit-2511 through the DiffSynth-Studio pipeline |

### `utils/prompts/` — the paper's prompts

| File or group | Purpose |
|---|---|
| `prompt_loader.py` | UTF-8 prompt loading and safe history-slot trimming |
| `paper_prompts.py` | named access to the model prompts |
| `judge_prompts.py` | named access to the judge prompts |
| `model/agent_planner_*` | online agent prompt |
| `model/code2world_*`, `gworld_*`, `mobileworld_*`, `html_wm_*` | Markov and history prompts for the HTML models |
| `model/closed_image_generation.md`, `flux_mobileworld_diffusion.md`, `qwen_image_edit.md` | direct image-model prompts |
| `judge/s_*`, `traj_user.md` | every judge prompt |
| `README.md` | index mapping prompts to the paper's figures |

The prompts for every model evaluated in the paper are included, even where the corresponding
adapter is not part of this release, so that the appendix figures can be checked against the exact
text that was used.

## `offline/` — offline track

| File | Purpose |
|---|---|
| `data.py` | data path constants and loading (samples jsonl, reference trajectory, ground-truth image derivation) |
| `rollout.py` | loads the action sequences, generates autoregressive predictions, writes run and summary records |
| `sharding.py` | round-robin sharding and deterministic merge for rollout |
| `cli.py` | evaluation CLI: sample selection, image preflight, parallel evaluation, aggregation |
| `scoring.py` | evaluation signature, result cache, ten-metric scoring, and aggregation |
| `judges.py` | step and trajectory judge requests, strict parsing, image encoding, and the VLM call |
| `visual_similarity.py` | SigLIP and DINOv2 similarity backends (`S_sig` / `S_dino`) |

## `online/` — online track

| File | Purpose |
|---|---|
| `rollout.py` | runs the GUI agent in closed loop and saves the plan, raw response, and structured failure reason at each step |
| `sharding.py` | round-robin sharding and deterministic merge for online rollout |
| `agent.py` | single-stage planner, plan validation, and per-attempt raw-response evidence |
| `actions.py` | conversion between planner plans and dataset semantic actions |
| `trajectory.py` | task loading and rollout reconstruction |
| `cli.py` | evaluation CLI, resume, and result writing |
| `scoring.py` | task signature, judge cache, per-task six-metric execution, and aggregation |
| `judges.py` | online step, trajectory, and milestone judges, image encoding, and the VLM call |

## `scripts/` — entry points

| File | Purpose |
|---|---|
| `env.sh` | shared shell environment loader: reads `utils/configs/paths.env`, resolves `PYTHON_BIN` / `VLLM_BIN` |
| `serve.sh <model>` | starts a local vLLM service for one world model |
| `run_all_offline.sh`, `run_all_online.sh` | iterate the configured rows, with evaluation when `EVALUATE_AFTER_RUN=1` |
| `run_vllm_job.sh` | brings up the required vLLM service, checks the endpoint and preflight, then runs one model/setting inside a single GPU job |
| `preflight.sh` | data validation, syntax compilation, unit tests, shell syntax, and optional per-model runtime checks |
| `validate_data.py` | structure, count, and image integrity checks for the offline and online data |
| `collect_results.py` | aggregates the complete offline and online result matrices into JSON and CSV |
| `compare_subset.py` | side-by-side scores, failure counts, and token usage for one `--subset N`, extrapolated to the full run |
| `run_offline_local_metrics.py` | computes `S_sig` / `S_dino` from an existing prediction tree without any API call |
| `vllm_ipc_tmpdir.sh` | prepares a short TMPDIR for the vLLM/ZeroMQ Unix socket |

## `tests/`

| File | Purpose |
|---|---|
| `test_adapters.py` | coordinates, prompts, renderer failures, and caching |
| `test_config.py` | the model matrix, model parameters, judge identity, paths, and renderer identity |
| `test_failure_policy.py` | failure classification, the fixed-denominator policy, and rollout resume |
| `test_offline_evaluate.py` | offline metric formulas, strict parsing, caching, partial mode, and signatures |
| `test_offline_sharding.py` | offline round-robin sharding and deterministic merge |
| `test_online_agent.py` | plan validation and unsupported agent actions |
| `test_online_evaluate.py` | online judges, 25-frame handling, caching, signatures, and aggregation |
| `test_online_pipeline.py` | portable paths, rollout reconstruction, task integrity, and hard failures |
| `test_online_sharding.py` | online round-robin sharding and deterministic merge |
| `test_project_layout.py` | dataset size and the configured model matrix |
| `test_model_failure_preflight.py` | how model-failure samples are handled at preflight and evaluation entry |
| `test_review_fixes.py` | regression tests for historical fixes |
| `test_subset.py` | the evenly spaced subset rule and its nesting property |

Tests that need the dataset skip themselves with a message when `data/` has not been downloaded.

## Minimal dependency sets

**Offline rollout** needs `utils/` (config, io, adapters, prompts, configs),
`data/offline_samples.jsonl` with the runtime sample files, `offline/rollout.py`,
`offline/sharding.py`, and `offline/data.py`. Reference *after* images are not rollout inputs.

`utils/adapters/registry.py` imports adapters lazily, but `utils/prompts/paper_prompts.py` loads the
prompt constants eagerly, so deployments should keep `utils/prompts/` whole rather than copying
individual Markdown files.

**Offline evaluation** additionally needs `offline/cli.py`, `offline/scoring.py`,
`offline/judges.py`, `offline/visual_similarity.py`, the generated prediction and run metadata, and
all reference trajectory images.

**Online rollout** needs `data/online_samples.jsonl`, all matching initial screenshots, `utils/`,
`online/rollout.py`, `online/agent.py`, `online/actions.py`, and `online/trajectory.py`.

**Online evaluation** additionally needs `online/cli.py`, `online/scoring.py`, `online/judges.py`,
a finished rollout directory, and the same task definitions. Sharded runs also need
`online/sharding.py`.
