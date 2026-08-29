"""Fixed subset for small-scale trial runs.

API models (the judge, the planner, closed world models) are billed per call, so it pays to
look at 10 or 100 samples before committing to the full 500 / 200. The subset depends only on
n, never on the model, so every model is tried on the same samples and the results are directly
comparable.
"""
from __future__ import annotations


def subset_ids(all_ids: list[str], n: int) -> list[str]:
    """按样本顺序等间距取 n 个：位置 0, N/n, 2N/n, ...。

    不取"前 n 个"，因为 offline 样本按类别分块排序（前 120 个全是 General_Tool），
    等间距才能覆盖各类别。n=10 是 n=100 的子集（offline 500 / online 200 均成立），
    小试跑的 rollout 与 judge 结果在大试跑里原样复用。
    """
    total = len(all_ids)
    if not 1 <= n <= total:
        raise ValueError(f"--subset 必须位于 1 到 {total} 之间，实际为 {n}")
    stride = total / n
    return [all_ids[int(index * stride)] for index in range(n)]
