You are evaluating the visual quality and usability of ONE predicted mobile GUI screenshot from a world model.

Evaluate only the screenshot itself. Do NOT judge whether it is the correct app, correct page, correct action result, or correct task state. Do NOT compare against ground truth. A wrong-but-clean UI can score high here; action/task correctness is evaluated by other metrics.

Score five binary criteria. Each criterion is 1 only if mostly satisfied with visible evidence, otherwise 0. The final score MUST equal the sum of the five criteria.

C1_valid_mobile_gui:
The image looks like a coherent mobile GUI screenshot, launcher, app screen, webview, dialog, keyboard, loading/splash screen, or system screen. Score 0 for blank/noise/random photo/non-UI images or a screen so malformed that the GUI state cannot be identified.

C2_render_integrity:
The screen is substantially complete and not visually broken. Score 0 for large white/unfilled holes, broken masks, repeated pasted regions, severe cropping, collapsed layout, or obvious rendering corruption. Small artifacts are allowed. Do not fail this criterion for one or two small local glitches if the main UI regions are still complete and readable.

C3_text_legibility:
Important visible text and labels are readable enough to understand the UI state. Score 0 for widespread gibberish, pseudo-text, heavy blur, repeated nonsense characters, or text so small/smeared that the main content cannot be read. If the screen naturally contains little text, score based on whether the available labels/status text are readable.

C4_component_coherence:
UI components such as buttons, cards, lists, tabs, search bars, toggles, icons, and navigation bars have coherent shapes, alignment, hierarchy, and spacing. Score 0 if components are melted, overlapping, floating randomly, duplicated unnaturally, or inconsistent with a usable interface. Minor imperfections in a small number of components should not fail this criterion if the overall UI component system remains coherent.

C5_interaction_readiness:
The screenshot is clear enough that a user or agent could understand the current GUI state and continue interacting when appropriate. Clean loading, splash, empty-state, or confirmation screens can score 1 if their state is visually understandable. Score 0 if the screen is too generic, unreadable, distorted, masked, or ambiguous to support the next interaction. Do not fail this criterion for task/action incorrectness or for minor local UI defects; fail it only when the visual state itself is not understandable or not usable.

Return strict JSON only:
{
  "C1_valid_mobile_gui": 0 or 1,
  "C2_render_integrity": 0 or 1,
  "C3_text_legibility": 0 or 1,
  "C4_component_coherence": 0 or 1,
  "C5_interaction_readiness": 0 or 1,
  "score": <integer 0-5>,
  "failure_modes": [...],
  "reasoning": "short evidence-based explanation"
}
