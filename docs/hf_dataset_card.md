---
license: cc-by-4.0
language:
- en
task_categories:
- image-to-image
- visual-question-answering
tags:
- gui
- gui-agent
- world-model
- mobile
- benchmark
size_categories:
- n<1K
configs:
- config_name: offline
  data_files: offline_samples.jsonl
- config_name: online
  data_files: online_samples.jsonl
---

# GUI-CC

GUI-CC is a benchmark for evaluating the **contextual consistency** of GUI world models when
they are used as *agent environments* rather than as isolated next-screen predictors.
A GUI world model predicts the next interface given the current screenshot and an action.
When that prediction is fed back as the next state, the rollout must stay coherent: app identity,
navigation history, created entities, selected options, and action affordances all have to remain
mutually consistent. GUI-CC measures exactly that.

This repository contains **only the evaluation data**. Code, prompts, and the evaluation harness
live in the GitHub repository linked below.

## Two tracks

| Track | Size | What it tests |
|---|---|---|
| **Offline reference-action** | 500 trajectories, 4,905 transitions | The model rolls out autoregressively along a fixed reference action sequence. Reference screenshots are used only for scoring, never as rollout inputs. |
| **Online agent-loop** | 200 tasks | A frozen probing GUI agent acts on model-generated screens until task completion, unusable output, or the step budget runs out. |

## Files

```
offline_samples.jsonl        500 lines, one offline sample per line (line order = sample order)
offline_data/<001..500>/
    reference_trajectory.jsonl   one line per step
    initial.png                  frame 0
    step_XXX_after.png           the screen after step XXX
online_samples.jsonl         200 lines, one task per line (line order = task order)
online_data/<001..200>/
    initial.png                  the task's frame 0
```

6,107 files in total, about 3.7 GB. The *before* frame of a step is not stored separately: step 0's
before frame is `initial.png`, and every later step's before frame is the previous step's after
frame (byte-identical upstream).

## Schema

**`offline_samples.jsonl`**

| Field | Meaning |
|---|---|
| `sample_id` | directory name, `001`–`500` |
| `task_instruction` | task instruction, the input to rollout and evaluation |
| `task_meta` | one-line summary of the task |
| `n_steps` | number of steps, equal to the number of trajectory lines |
| `source_dataset`, `source_episode_id` | upstream GUI-Odyssey dataset and episode id |
| `source_category`, `source_apps` | upstream task category and the apps involved |

**`reference_trajectory.jsonl`** (one line per step)

| Field | Meaning | Present for |
|---|---|---|
| `step_id` | 0-based, matches `step_{step_id:03d}_after.png` | all |
| `semantic_action.type` | `tap` / `scroll` / `long_press` / `navigate_home` / `navigate_back` | all |
| `semantic_action.target` | text description of the target element | all |
| `semantic_action.low_level_instruction` | low-level instruction for the step | tap / scroll / long_press |
| `semantic_action.source_coord`, `source_coord_norm` | tap point in pixels and in per-mille | tap / long_press |
| `semantic_action.grounding.bbox` | bounding box of the target element | tap / long_press |
| `semantic_action.direction`, `start_norm`, `end_norm` | swipe direction and endpoints | scroll |

**`online_samples.jsonl`**

| Field | Meaning |
|---|---|
| `task_id` | directory name, `001`–`200` |
| `source_task_name`, `template_id` | descriptive name used during collection; task template (18 in total) |
| `category`, `apps` | task-level label; apps involved |
| `step_budget` | maximum number of agent decisions (a terminal action counts as one) |
| `instruction` | task instruction |
| `initial_state` | text description of the initial state |
| `milestones` | ordered list of `{id, type, assertion}`; `type` is a list drawn from `progress`, `memory_persistence`, `terminal_success`; the last milestone always contains `terminal_success` |

Note that `initial.png` is a concrete first GUI screenshot, **not** a random seed.

## Statistics

| Statistic | Offline | Online |
|---|---:|---:|
| Tasks | 500 | 200 |
| Transitions | 4,905 | -- |
| Avg. steps / step budget | 9.81 (7–14) | 15.74 (6–24) |
| Apps | 130 | 30 |
| App categories | 6 | 6 |
| Task templates | -- | 18 |
| Avg. milestones | -- | 3.97 (2–5) |

Offline screenshots come from an Android 14 (SDK 34) emulator at four device resolutions:
720×1280 (176 samples), 1080×2400 (124), 1344×2992 (109), 1440×3120 (91).
All online screenshots are 1080×2400.

The online `category` field takes seven values. Six of them are app domains (Communication 27,
Finance & Business 19, Media & Entertainment 41, Shopping & Delivery 24, System & Utility 41,
Travel & Navigation 33); the seventh, `Cross-App` (15), is a task-level label for cross-app
workflows and is not a seventh app domain.

## Usage

```python
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="minuzero/GUI-CC", repo_type="dataset")
```

Then point the evaluation harness at that directory. Full rollout and evaluation instructions are in
the GitHub repository.

## Provenance and terms of use

**Offline track.** All 500 trajectories are derived from
[GUI-Odyssey](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey), which is released under
CC BY 4.0. We filtered, normalized, and re-annotated the upstream episodes; coordinates were
restored from the normalized 0–1000 range to pixels, and raw action records were converted into the
semantic action schema above. GUI-CC is released under the same license, and we retain the upstream
`source_dataset` and `source_episode_id` fields so that every sample can be traced back.

**Online track.** The 200 initial screenshots were captured by the authors in an Android emulator
across 30 real applications. The applications, their interfaces, trademarks, and displayed content
remain the property of their respective owners. These screenshots are distributed **for
non-commercial research use only**, under fair use for academic evaluation. If you are a rights
holder and would like a screenshot removed, please open an issue on the GitHub repository or
contact the corresponding authors.

This dataset is intended for research on GUI world models and GUI agents. We oppose any use of it
to build systems that deceive users, bypass safety controls, or interact with services without
authorization.

## Citation

```bibtex
@inproceedings{fu2026guicc,
  title     = {{GUI-CC}: Benchmarking Contextual Consistency of {GUI} World Models as Agent Environments},
  author    = {Fu, Lin and Yang, Zheyuan and Zhang, Tianhui and Wei, Jinbiao and
               Gan, Guo and Liu, Boxu and Zhao, Yilun and Rong, Yu},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

Please also cite GUI-Odyssey, from which the offline track is derived:

```bibtex
@article{lu2024guiodyssey,
  title   = {GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices},
  author  = {Lu, Quanfeng and Shao, Wenqi and Liu, Zitao and Meng, Fanqing and Lin, Boxuan and
             Chen, Yaxin and Huang, Botong and Zhang, Kaipeng and Qiao, Yu and Luo, Ping},
  journal = {arXiv preprint arXiv:2406.08451},
  year    = {2024}
}
```
