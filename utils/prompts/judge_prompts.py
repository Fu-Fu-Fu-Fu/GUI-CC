"""VLM judge prompts. Constant names map one-to-one onto the paper's metric names."""
from __future__ import annotations

from .prompt_loader import load_prompt

# 逐步指标：judge 看一对 before/after 截图。
S_AD_SYSTEM_PROMPT = load_prompt("judge/s_ad_system.md")
S_AD_USER_TEMPLATE = load_prompt("judge/s_ad_user.md")
S_ID_SYSTEM_PROMPT = load_prompt("judge/s_id_system.md")
S_ID_USER_PROMPT = load_prompt("judge/s_id_user.md")
# judge 是自由文本模型，输出需要落在 s_id_system.md 声明的枚举内。
S_ID_CATEGORIES = frozenset({
    "tap", "long_press", "scroll", "type_text", "open_app",
    "navigate_home", "navigate_back", "wait", "none",
})
S_ELE_LAY_SYSTEM_PROMPT = load_prompt("judge/s_ele_lay_system.md")
S_ELE_LAY_USER_PROMPT = load_prompt("judge/s_ele_lay_user.md")
S_USE_SYSTEM_PROMPT = load_prompt("judge/s_use_system.md")
S_USE_USER_PROMPT = load_prompt("judge/s_use_user.md")

# 轨迹级指标：judge 看整条预测轨迹，共用同一个 user 模板。
TRAJ_USER_TEMPLATE = load_prompt("judge/traj_user.md")
S_CP_SYSTEM_PROMPT = load_prompt("judge/s_cp_system.md")
S_RD_SYSTEM_PROMPT = load_prompt("judge/s_rd_system.md")

# 轨迹级指标中 offline 与 online 各有一个：S_rap 对参考轨迹，S_mp 对任务目标。
S_RAP_SYSTEM_PROMPT = load_prompt("judge/s_rap_system.md")
S_RAP_USER_TEMPLATE = load_prompt("judge/s_rap_user.md")
S_MP_SYSTEM_PROMPT = load_prompt("judge/s_mp_system.md")
S_MP_USER_TEMPLATE = load_prompt("judge/s_mp_user.md")
