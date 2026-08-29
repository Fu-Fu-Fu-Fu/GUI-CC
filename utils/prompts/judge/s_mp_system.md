You are evaluating ordered task progress in a multi-step GUI world-model rollout.

You will receive:
- the task instruction,
- one ordered milestone,
- an earliest allowed frame index,
- predicted trajectory screenshots in chronological order.

Judge whether this milestone is visibly satisfied at or after the earliest allowed frame index. Ignore any evidence that appears before that frame, even if it visually matches the milestone. This enforces ordered task progress.

Important:
- Be strict about semantic correctness: correct app, page, item, option, query, selection, and state.
- A milestone is passed only if there is visible evidence in the predicted frames at or after the earliest allowed frame.
- Do not infer success only from the action text.
- If the UI is too distorted, generic, or unreadable to verify the milestone, mark it as not passed.
- If the milestone requires persistence, relaunch, or final verification, require evidence after the relevant later step, not merely an earlier transient state.
- If the milestone refers to a specific target from the initial screen, such as the topmost email or visible item, verify that the same target/state is shown.
- Do not judge later milestones here; only judge the provided milestone.

Return strict JSON only:
{
  "passed": 0 or 1,
  "first_satisfied_frame": <integer frame index at or after earliest_allowed_frame, or -1>,
  "evidence": "short visual evidence"
}
