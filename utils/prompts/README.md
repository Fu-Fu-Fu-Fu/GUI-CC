# GUI-CC Prompt Templates

This directory holds the prompt templates from the paper's appendices, revised through a code
audit. They are not a verbatim transcription of the submitted PDF; behavior-affecting revisions are
recorded in the repository history and carried into the next version of the paper.

## Layout

- `model/`: Appendix A, model input prompts. File names are semantic; the figure numbers appear
  only in the index below.
- `judge/`: Appendix B, VLM judge prompts.
- `prompt_loader.py`: the small loader used at runtime.
- `paper_prompts.py`: exports the model input prompts under named constants.
- `judge_prompts.py`: exports the judge prompts; the constant names map one-to-one onto the
  paper's metric names.

Only the prompts currently in use are kept here; earlier versions are not shipped.

Image placeholders such as `[image]`, `[annotated_image]`, and `[image_step0]` are carried at
runtime by OpenAI-compatible multimodal message parts. The template files keep the text and the
code attaches the images explicitly.

The history user templates for Figures 5, 7, 8, and 10 are both the paper template and the runtime
text source; the code only replaces the image placeholders with the corresponding multimodal image
parts.

Prompts are included for every model evaluated in the paper, including models whose adapter is not
part of this release, so that the appendix figures can be checked against the exact text used.

## Index

- Figure 3: single-stage GUI agent baseline (emits the action and pixel coordinates directly)
  - `model/agent_planner_system.md`
  - `model/agent_planner_user.md`
- Figure 4: Code2World without history
  - `model/code2world_system.md`
  - `model/code2world_user.md`
- Figure 5: Code2World with history
  - `model/code2world_history_system_suffix.md`
  - `model/code2world_history_user.md`
- Figure 6: gWorld without history
  - `model/gworld_user.md`
- Figure 7: gWorld with history
  - `model/gworld_history_prefix.md`
  - `model/gworld_history_user.md`
- Figure 8: MobileWorld-html with and without history
  - `model/mobileworld_system.md`
  - `model/mobileworld_user.md`
  - `model/mobileworld_history_system_suffix.md`
  - `model/mobileworld_history_user.md`
- Figure 9: closed-model HTML generation without history
  - `model/html_wm_system.md`
  - `model/html_wm_user.md`
- Figure 10: closed-model HTML generation with history
  - `model/html_wm_history_system_suffix.md`
  - `model/html_wm_history_user.md`
- Figure 11: closed image-generation models
  - `model/closed_image_generation.md`
- Figure 12: FLUX.2-dev / MobileWorld-Diffusion image generation
  - `model/flux_mobileworld_diffusion.md`
- Figure 13: Qwen-Image-Edit-2511 image generation
  - `model/qwen_image_edit.md`
- `S_ad` action adherence (per transition)
  - `judge/s_ad_system.md`
  - `judge/s_ad_user.md`
- `S_id` inverse-dynamics identifiability (per transition)
  - `judge/s_id_system.md`
  - `judge/s_id_user.md`
- `S_ele` / `S_lay` element and layout judging (per transition, one shared prompt pair)
  - `judge/s_ele_lay_system.md`
  - `judge/s_ele_lay_user.md`
- Figure 14: `S_use` GUI state usability judge
  - `judge/s_use_system.md`
  - `judge/s_use_user.md`
- Figure 15: `S_cp` state and context persistence judge
  - `judge/s_cp_system.md`
  - `judge/traj_user.md`
- Figure 16: `S_rd` action-controlled rollout dynamics judge
  - `judge/s_rd_system.md`
  - `judge/traj_user.md`
- Figure 17: `S_rap` reference action progress judge
  - `judge/s_rap_system.md`
  - `judge/s_rap_user.md`
- Figure 18: `S_mp` ordered milestone progress judge
  - `judge/s_mp_system.md`
  - `judge/s_mp_user.md`
