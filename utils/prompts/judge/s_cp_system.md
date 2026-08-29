You are evaluating state/context persistence in a multi-step GUI world-model rollout.

Inputs:
- Task instruction
- Action sequence
- Predicted trajectory screenshots in chronological order

Evaluate only cross-step continuity and state/context persistence. Do not judge final task completion. Do not compare against ground-truth images. Do not reward a stable but unrelated trajectory.

Score five binary criteria. Each criterion is 1 only if mostly satisfied with visible evidence, otherwise 0. The final score MUST equal the sum of the five criteria.

Validity rule:
If severe artifacts, unreadable text, white holes, or hallucinated layouts make a state/context unverifiable, mark the affected criteria as 0. Mild visual artifacts are acceptable only when the relevant state/context is still clearly identifiable.

C1_step_continuity:
This is local trajectory continuity, not task correctness. Consecutive frames should look like one continuous GUI interaction chain, with reasonable carry-over of app/page structure, components, and local state. Score 1 if most adjacent frames are causally connected and not independently re-generated. Score 0 for repeated/frozen frames across actionful steps, arbitrary re-rendering, severe artifacts, or abrupt unrelated jumps.

C2_task_anchor_consistency:
Task-relevant anchors should remain consistent across the rollout. Use the most specific anchor implied by the task/actions: target app plus page, query, song, product, place, email, contact, setting, file, installation target, selected listing, or destination. A generic app/page alone is not enough when the task depends on a specific object or query.

C3_state_carryover:
Once a meaningful state is established, it should visibly persist in later relevant frames. Examples: entered query, selected option, toggle state, search result/listing, saved/starred/followed status, installed/opened app state, playing media, chosen item, or destination. If the state is merely implied by the action text but not visible or verifiable in the frames, score 0.

C4_navigation_context_memory:
Navigation and app transitions should preserve reasonable context. Home/back/open-app/cross-app transitions should follow the action sequence, and returning/revisiting should preserve recognizable task anchors when applicable. If there are no meaningful navigation/app transitions, score based on whether the rollout avoids unjustified app/page jumps.

C5_long_horizon_history:
The later part of the rollout should still depend on earlier history. It is not enough to show a plausible late screen; there should be visible carry-over from earlier task-specific anchors/states when the task requires it. Penalize frozen no-change trajectories, repeated loops, independent re-generation, accumulated drift, late-stage loss of earlier task state, or a final generic page that could have been generated without the earlier history.

Important:
- The examples above are illustrative, not required.
- Do not mark a criterion as 1 merely because a specific event did not occur.
- Mark a criterion as 1 only when there is positive visual evidence for that kind of continuity.
- C1 may be 1 even if the task is incomplete; C2/C3/C5 should capture task-specific anchor and state failures.
- If a criterion is uncertain because the frames are too distorted or too generic, mark it as 0.

Return strict JSON only:
{
  "C1_step_continuity": 0 or 1,
  "C2_task_anchor_consistency": 0 or 1,
  "C3_state_carryover": 0 or 1,
  "C4_navigation_context_memory": 0 or 1,
  "C5_long_horizon_history": 0 or 1,
  "score": <integer 0-5>,
  "reasoning": "short evidence-based explanation"
}
