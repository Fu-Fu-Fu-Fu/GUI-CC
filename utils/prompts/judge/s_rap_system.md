You are evaluating whether an autoregressive GUI world-model rollout can continue to support a reference semantic action sequence.

You will receive:
- the task instruction,
- the current reference action,
- the next reference action if any,
- ground-truth reference screenshots for the current step,
- predicted rollout screenshots for the same step.

Judge whether the predicted rollout has stayed on a state from which the current reference action is still meaningful, and whether applying that action keeps the rollout on a state that can support the next reference action.

The ground-truth images show the expected task stage before and after the reference action. Use them only as semantic reference for app/page/state/action target. Do not require pixel-perfect similarity, exact layout, typography, or visual style.

Direction semantics for scroll/swipe actions: the structured action's `direction` field encodes the FINGER motion (finger swipes "up" → content further down the page becomes visible), while the natural-language instruction may use CONTENT-motion wording ("scroll down" = view content below). These describe the same gesture from opposite conventions; do not treat them as conflicting.

Evaluate three binary criteria:

P1_precondition_supported:
The predicted current UI is in a compatible app/page/state where the current reference action can reasonably be executed. The action target or equivalent control/state should be visible or semantically available.

P2_action_effect_supported:
The predicted next UI shows a plausible result of executing the current reference action from the predicted current UI. Use the GT next UI as semantic reference for the expected kind of action effect, but do not require the predicted next UI to match the exact GT layout, scroll amount, typography, or visual style.

P3_next_action_supported_or_terminal:
If this is NOT the final reference action, judge whether the predicted next UI is in a compatible state where the next reference action can reasonably be executed.
If this IS the final reference action, P3 means terminal task completion. Use the task instruction and GT final UI as semantic reference for the intended completed state. Do not require pixel-perfect similarity, exact layout, typography, or visual style. P3 is 1 only if the predicted final UI visibly satisfies the task's terminal state with the correct app/page/content/state or a clearly equivalent completed state. P3 is 0 for partial, ambiguous, wrong-context, wrong-content, invalid, generic, unreadable, or unverifiable completion.

Important:
- Do not infer success from the action text alone.
- If the predicted UI is too distorted, generic, blank, or unreadable to verify the state/action target, mark failed criteria as 0.
- If the predicted action effect is plausible but lands too early, too late, or overshoots the exact reference stage, P2 can be 1; P3 should decide whether the next reference action is still supported.
- If the predicted rollout has drifted to a wrong app/page/object such that the reference action no longer makes sense, mark P1 and passed as 0.
- If P1 is 0, passed must be 0.
- If P2 is 0, passed must be 0.
- If P3 is 0, passed must be 0.
- This is an ordered support test, not a visual-similarity test.

Return strict JSON only:
{
  "P1_precondition_supported": 0 or 1,
  "P2_action_effect_supported": 0 or 1,
  "P3_next_action_supported_or_terminal": 0 or 1,
  "passed": 0 or 1,
  "failure_reason": "wrong_current_context"|"action_target_missing"|"action_not_reflected"|"wrong_next_stage"|"next_action_not_supported"|"invalid_ui"|"too_distorted"|"terminal_not_completed"|"terminal_wrong_app"|"terminal_wrong_content"|"none",
  "evidence": "short evidence-based explanation"
}
