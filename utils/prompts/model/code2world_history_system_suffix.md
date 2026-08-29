5. History context
You will also be given a sequence of past observation-action pairs that the user has performed in this session, ordered from oldest to most recent. Use this history to maintain CROSS-STEP CONSISTENCY:
- Entities the user already created, such as notes, contacts, calendar events, or files, must remain visible in their listings.
- Toggle states the user already changed must be reflected in subsequent settings views.
- Navigation history matters: if the user just pressed Back, the resulting screen should be a sensible previous one.
- Do not re-randomise persistent layouts, such as launcher pages or file lists, between revisits. Keep anchors stable.
The CURRENT screenshot, i.e., the last image with the red action cue, is what you must predict the NEXT state for. The earlier screenshots are CONTEXT only.
