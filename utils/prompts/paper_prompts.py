from __future__ import annotations

from .prompt_loader import load_prompt

# Figure 3：GPT-5.5 单阶段 GUI agent 基线。
PLANNER_SYSTEM_PROMPT = load_prompt("model/agent_planner_system.md")
PLANNER_USER_TEMPLATE = load_prompt("model/agent_planner_user.md")

# Figure 4-5：Code2World。
CODE2WORLD_SYSTEM_PROMPT = load_prompt("model/code2world_system.md")
CODE2WORLD_USER_TEMPLATE = load_prompt("model/code2world_user.md")
HISTORY_SYSTEM_SUFFIX = "\n\n" + load_prompt("model/code2world_history_system_suffix.md")
CODE2WORLD_HISTORY_USER_TEMPLATE = load_prompt("model/code2world_history_user.md")

# Figure 6-7：gWorld。
GWORLD_USER_TEMPLATE = load_prompt("model/gworld_user.md")
GWORLD_HISTORY_PREFIX_TEMPLATE = load_prompt("model/gworld_history_prefix.md")
GWORLD_HISTORY_USER_TEMPLATE = load_prompt("model/gworld_history_user.md")

# Figure 8：MobileWorld-html。
MOBILEWORLD_SYSTEM_PROMPT = load_prompt("model/mobileworld_system.md")
MOBILEWORLD_USER_TEMPLATE = load_prompt("model/mobileworld_user.md")
MOBILEWORLD_HISTORY_SYSTEM_SUFFIX = " " + load_prompt("model/mobileworld_history_system_suffix.md")
MOBILEWORLD_HISTORY_USER_TEMPLATE = load_prompt("model/mobileworld_history_user.md")

# Figure 9-10：闭源 HTML 世界模型。
HTML_WM_SYSTEM_PROMPT = load_prompt("model/html_wm_system.md")
HTML_WM_USER_TEMPLATE = load_prompt("model/html_wm_user.md")
HTML_WM_HISTORY_SYSTEM_SUFFIX = "\n\n" + load_prompt("model/html_wm_history_system_suffix.md")
HTML_WM_HISTORY_USER_TEMPLATE = load_prompt("model/html_wm_history_user.md")

# Figure 11-13：图像生成世界模型。
CLOSED_IMAGE_GEN_PROMPT_TEMPLATE = load_prompt("model/closed_image_generation.md")
MOBILEWORLD_DIFFUSION_PROMPT_TEMPLATE = load_prompt("model/flux_mobileworld_diffusion.md")
QWEN_IMAGE_EDIT_PROMPT_TEMPLATE = load_prompt("model/qwen_image_edit.md")
