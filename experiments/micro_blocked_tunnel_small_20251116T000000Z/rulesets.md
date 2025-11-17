# Ruleset snippets (referenced by rules_id)

Use with environment variables:
- Structured: `LLMGRID_STRUCTURED_EXTRA_RULES`
- Freeform: `LLMGRID_FREEFORM_EXTRA_RULES`

Each ruleset is line-based; non-empty lines are appended to the strategy block.

## none_default
No extra rules (radio disabled).

## safety_bias
- If last_move_outcome != OK, STAY for one turn before any new MOVE.
- Prefer unexplored cells over backtracking when safe; if unsure, STAY.

## ff_default
No extra lines; rely on base freeform rules.

## ff_cautious
- Send CHAT only if contended_neighbors != 0 or last_move_outcome != OK.
- If you cannot argue a move is safe, STAY and explain.

## ff_tag_light
- CHAT format: [TAG payload] short note. TAG in {INTENT, YIELD, INFO}.
- If you plan to enter a choke or contested cell, send [INTENT MOVE_dir].
- If you choose to STAY in a choke, send [YIELD cell=(x,y)].
- Receivers: honor YIELD for 1 turn; if lower id sent INTENT, yield.

## ff_tag_strict
- CHAT format: [TAG payload] short note. TAG in {INTENT, YIELD, INFO, ROUTE}.
- Always send [INTENT MOVE_dir] before entering the main corridor.
- If two agents target same corridor, lower id sends INTENT, higher id STAYs 2 turns.
- Honor any YIELD for 2 turns unless goal is adjacent and free.

## ff_kv_light
- CHAT as key=value: KIND=INTENT|YIELD|INFO; ACTION=MOVE_X or STAY; CELL=(x,y); NOTE=free text.
- Send only when contended_neighbors != 0 or after a block.
- Receivers: if KIND=YIELD and CELL matches your target, STAY 1 turn.

## ff_kv_strict
- CHAT key=value as above, but required before moving into any one-cell choke.
- If you send KIND=INTENT, include CELL target; lower id gets priority.
- Receivers: if higher id sees lower id INTENT for same CELL, STAY 2 turns.

## struct_intent_only
- Use INTENT only; no REQUEST. Send INTENT before entering corridor cells.
- If last_move_outcome != OK, STAY then resend INTENT with updated target.

## struct_yield_rule
- Lowest agent_id has right of way in chokes; higher ids send REQUEST:YIELD and STAY 1 turn.
- If you receive REQUEST:YIELD, STAY unless you are lower id and already in the cell.

## struct_sparse_cautious
- Send INTENT or REQUEST only after a block or contended_neighbors != 0.
- After sending a REQUEST, STAY until the corridor is free or 2 turns have passed.

## ff_priority_lowest_goes
- Lower id always proceeds in a choke; higher id yields without sending unless ambiguous.
- Only CHAT when divergence from this rule is necessary; keep messages ≤80 chars.
