"""Attribute a failure to the model or to the infrastructure.

A model failure (unparseable output, empty output, HTML the model wrote that will not
render) scores 0 and stays in the fixed denominator. An infrastructure failure (timeout,
OOM, browser crash) blocks evaluation instead and has to be fixed and rerun.
"""
from __future__ import annotations

INFRA_KEYWORDS = (
    "timeout", "timed out", "connection", "connect", "503", "504", "502",
    "500", "5xx", "oom", "out of memory", "cuda", "gpu", "chromium",
    "browser", "playwright", "target closed", "navigation", "socket",
    "dns", "refused", "reset by peer", "broken pipe", "eof", "unreachable",
)


def classify_failure(error, stage):
    error_lower = (error or "").lower()
    if stage == "parse":
        return {"class": "model", "kind": stage, "message": error}
    if stage == "generation":
        if any(k in error_lower for k in INFRA_KEYWORDS):
            return {"class": "infrastructure", "kind": "generation_infra", "message": error}
        return {"class": "model", "kind": stage, "message": error}
    if stage == "render":
        # 浏览器侧的故障一律算基础设施：阻塞聚合、修好重跑。
        # 超时也在此列。超时既可能来自模型写的 HTML，也可能来自机器负载
        # （三个 vLLM 服务 + 三路并发 rollout），事后无法分辨；判成模型失败会让
        # 机器忙时的抖动压低某个 baseline 的分数，而这个错误不可恢复。
        # 判成基础设施的代价只是重跑一次，可恢复，所以取这一侧。
        if any(k in error_lower for k in (
                "chromium", "browser", "playwright", "target closed", "navigation",
                "protocol error", "capture screenshot", "timeout", "timed out")):
            return {"class": "infrastructure", "kind": "render_infra", "message": error}
        return {"class": "model", "kind": "render", "message": error}
    if stage == "agent":
        # agent（planner）是所有被测世界模型共用的固定组件。它自己失败
        # （API 报错、重试用尽仍不调工具、网关返回了别的模型）不是世界模型的
        # 问题，记成 model failure 会让坏掉的 agent 产出一张看似完整的表。
        return {"class": "infrastructure", "kind": "agent", "message": error}
    if stage == "request":
        if "empty_completion" in error_lower:
            return {"class": "model", "kind": "empty_completion", "message": error}
        return {"class": "infrastructure", "kind": "request", "message": error}
    if any(k in error_lower for k in INFRA_KEYWORDS):
        return {"class": "infrastructure", "kind": stage, "message": error}
    return {"class": "model", "kind": stage, "message": error}
