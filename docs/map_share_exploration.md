# Map-Sharing Exploration (design sketch)

**Goal:** let agents share small, structured snippets of their explored map to improve coordination in chokepoints, without changing physics or adding a safety veto.

## Minimal viable approach
- Keep existing message schema; use `Chat` messages with a constrained format:
  - `MAP row=y rng=x1-x2 walls=... goals=...`
  - Fit under 96 chars; send only when you uncover new corridors near a choke.
- Receiver behavior (prompt rules only):
  - Update your mental map: treat shared walls/goals as known.
  - Prefer unexplored cells consistent with shared info; avoid entering reported walls.
- No code changes required if we confine to prompt rules; evaluate via a dedicated baseline.

## Medium approach (light code change)
- Add a new message type `MsgMap` (kind="MAP", payload string) to `OutgoingMessage` union.
- Allow it for freeform (or a new `comm_strategy=map_share`).
- Keep payload tiny: run-length of walls in a 3x3 window around the sender, plus sender abs coords.
- Receiver: still prompt-only (parse the payload), no environment-side map merge.

## Ambitious approach (map merge)
- Add a `map_delta` structure to messages and merge into `AgentMap` on receipt.
- Needs code:
  - `AgentMap.export_delta(patch)` and `AgentMap.apply_delta(delta)`.
  - New message type carrying delta; env merges on receipt.
- Higher risk; require careful size limits and security checks.

## When agents meet: face-to-face sharing plan
- Default behavior: if two agents occupy the same or adjacent cell, each must broadcast one `MAP` snippet on that turn, even without an explicit request.
- Request channel (prompt-only first):
  - Add a chat verb `REQMAP x=y` that asks the peer for a local map slice around its current coordinates.
  - Rule: if you receive `REQMAP`, reply on your next turn with `MAP row=y rng=x1-x2 walls=... goals=...` or say `MAP none` if you have nothing new.
- Snippet content:
  - 3x3 or 5x3 window centered on sender coordinates (pick 3x3 for the minimal version).
  - Include absolute sender coords, known goals in the window, and walls as run-length pairs.
  - Keep under 96 chars; one `MAP` per turn even after a request.
- Enforcement against silent agents:
  - Prompt rule: failing to answer `REQMAP` is considered non-cooperative; prioritize answering before exploration.
  - Add a counter in eval to flag unanswered requests; treat repeated misses as a run failure mode.
  - Optionally add a lightweight self-check: if you sent a `REQMAP` last turn and did not see a `MAP`, send a reminder once then continue exploring.
- Medium/ambitious code hooks (if prompt rules are too weak):
  - Allow a new message kind `MAP_REQ` and `MAP_SNIP` in `OutgoingMessage`.
  - Receiver must echo an ack `MAP_ACK` to track compliance; the environment can log missing acks.
  - Add instrumentation: counts of requests made, requests answered, requests ignored.

## Experiments to run (cheap)
1) Prompt-only MAP chat baseline:
   - New ruleset: send MAP chat when you reveal a new corridor adjacent to a choke; include coords of free cell and any walls.
   - Compare collisions, success, coverage vs freeform_default on micro_blocked_tunnel_small.
2) Structured MAP (MsgMap) baseline:
   - Add MsgMap to schema; allow one MAP per turn; payload = sender (x,y) + 3x3 walls bitmap.
   - Receiver prompted to honor MAP info when planning.

## Guards
- Keep messages ≤96 chars; one MAP per turn.
- Do not change physics; collisions remain real.
- Measure: success, collisions, coverage, loops; count MAP messages and their helpful/harmful effect like other comm.

## Next steps
- Agree on scope (prompt-only vs MsgMap).
- Implement chosen scope on this `map-share-prototype` branch.
- Add baselines and run on micro_blocked_tunnel_small (2 agents, 50 turns).
