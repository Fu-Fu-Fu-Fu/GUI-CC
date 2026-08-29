You are evaluating action-controlled temporal dynamics in a multi-step GUI world-model rollout.

Inputs:
- Task instruction
- Action sequence
- Predicted trajectory screenshots in chronological order

Evaluate whether the predicted GUI changes are controlled by the action sequence. Focus on whether actions produce synchronized and plausible visual changes over time. Do not judge final task completion. Do not compare against ground-truth images. Do not reward a stable but frozen trajectory.

Score five binary criteria. Each criterion is 1 only if mostly satisfied with visible evidence, otherwise 0. The final score MUST equal the sum of the five criteria.

C1_action_responsiveness:
Most actionful steps should produce an appropriate visible response. Taps should open/select/focus/toggle when expected, type actions should visibly affect input or text state, scroll actions should move content, and home/back/open-app actions should change navigation context. Score 0 if the rollout is mostly unchanged despite actionful steps.

C2_action_change_synchronization:
Visible changes should happen immediately after the corresponding action, not one or more steps late, early, or independently of the action. Score 0 if changes appear at arbitrary times, if the same action repeatedly has no effect and then an unrelated jump occurs, or if frames change without a matching action.

C3_transition_order_coherence:
The trajectory should progress in the same order as the action sequence, without unexplained resets, skipped intermediate contexts, spontaneous app/page switches, or impossible jumps. Score 1 if most transitions form a coherent temporal chain.

C4_change_scope_control:
Changes should be controlled in scope. Local actions should not cause arbitrary full-screen re-generation, brand/app identity changes, unrelated content replacement, or large layout reshuffling unless the action is navigation/home/open-app/back/page transition. Score 0 if unrelated areas frequently drift or the whole UI is resampled without cause.

C5_no_freeze_or_temporal_degradation:
The rollout should remain usable over the horizon. Score 0 if it becomes frozen/repetitive after actionful steps, falls into loops, accumulates artifacts, develops white holes, has increasingly corrupted text/icons, collapses layout, or drifts into generic/unrelated screens. A visually clean but no-change/frozen trajectory MUST score 0 for this criterion.

Important:
- Explicit wait/status/no-op steps may remain stable.
- Large changes are allowed for home, back, open-app, page navigation, or dialog transitions.
- C1/C2 are about action-response timing and presence; C3/C4/C5 are about temporal control and stability across the whole rollout.
- Mild visual artifacts are acceptable only if action-controlled dynamics remain clear.
- If a criterion is uncertain because frames are too distorted or too generic, mark it as 0.

Return strict JSON only:
{
  "C1_action_responsiveness": 0 or 1,
  "C2_action_change_synchronization": 0 or 1,
  "C3_transition_order_coherence": 0 or 1,
  "C4_change_scope_control": 0 or 1,
  "C5_no_freeze_or_temporal_degradation": 0 or 1,
  "score": <integer 0-5>,
  "reasoning": "short evidence-based explanation"
}
