Task instruction:
{instruction}

Step:
{step_index} of {total_steps}

Current reference action:
{action_desc}

Current reference action JSON:
{action_json}

Next reference action:
{next_action_desc}
{terminal_block}

Image definitions:
Image 1: GT current UI before the reference action.
Image 2: GT next UI after the reference action. For the final step, this is the GT final UI.
Image 3: Predicted current UI in the autoregressive rollout.
Image 4: Predicted next UI after the current reference action.

Does the predicted rollout still support this reference action step?
