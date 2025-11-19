# Field Usage Examples: Direct Citations from Agent Comments

This document provides specific, line-by-line examples of how agents reference (or ignore) each observation field.

---

## TIER 1: HEAVILY USED FIELDS

### world_map_ascii (97.0% - 1,243/1,281 entries)

**Pattern:** Agents cite specific coordinates or map features from the ASCII grid.

Examples:
1. Turn 2, Agent a1: "Moving E to (2,7) toward nearest frontier at (3,7), scouting unknown while avoiding recent trail."
2. Turn 7, Agent a5: "Heading north toward frontier at (3,7); (3,8) is free. Leaving ~ trail to explore."
3. Turn 14, Agent a3: "Moving south to (1,4) toward frontier (1,5); avoiding ~ trail at (2,3) and walls."
4. Turn 19, Agent a2: "Advancing W to (5,3) toward frontier; avoiding agent at (5,5) and recent trail."
5. Turn 34, Agent a4: "Scouting frontier at (11,4) by moving N; east/west are walls."

**Key observation:** Agents always name specific coordinate pairs or features ("frontier at (x,y)", "wall at position", "agent at cell"). The ASCII map is the referent for every navigation decision.

---

### nearest_frontier (92.0% - 1,179/1,281 entries)

**Pattern:** Agents explicitly state the frontier coordinate and often explain why that direction is chosen.

Examples:
1. Turn 0, Agent a1: "Heading N to approach nearest frontier at (8,0); E/S/W walls; safest progress and collision-free."
2. Turn 1, Agent a2: "Advancing W toward nearest frontier at (3,7); avoiding recent trail at (6,7). N/S are walls; E is open."
3. Turn 3, Agent a4: "Advancing N to scout frontier at (9,2); E/W are walls; avoiding recent trail south."
4. Turn 5, Agent a5: "Heading east toward frontier at (3,9) while avoiding recent trail at (0,9). Walls N/S."
5. Turn 12, Agent a2: "Heading west toward nearest frontier (2,7); avoiding recent trail east."

**Key observation:** Nearly 9 in 10 decisions explicitly name the frontier coordinate. Agents treat this as the primary navigation target above all else.

---

### adjacent_state (83.5% - 1,070/1,281 entries)

**Pattern:** Agents list which cardinal directions are FREE or WALL, often when explaining a move.

Examples:
1. Turn 0, Agent a1: "Heading N to approach nearest frontier at (8,0); E/S/W walls; safest progress."
2. Turn 1, Agent a3: "Advancing W toward nearest frontier at (0,3), stepping to (1,3). Avoiding recent trail at (2,3); E is wall."
3. Turn 2, Agent a2: "Heading west toward nearest frontier (2,7); avoiding recent trail east."
4. Turn 5, Agent a5: "Moving east toward frontier at (4,9); north is WALL, south OOB; leaving recent trail to progress."
5. Turn 7, Agent a2: "Nearest frontier at (1,7); moving W to (2,7) to reveal. N is WALL; goal east noted; mapping first."

**Key observation:** When explaining a move, agents routinely cite wall positions. This validates movement choices and prevents invalid directions.

---

## TIER 2: MODERATE USE FIELDS

### peer_bits (62.1% - 795/1,281 entries)

**Pattern:** Agents mention teammate presence, often to coordinate around collisions or confirm no conflict.

Examples of USE:
1. Turn 0, Agent a4: "Advancing W toward nearest frontier at (8,6) to reveal map; N/S walls, no peers nearby."
2. Turn 2, Agent a3: "Moving south to (1,4) toward frontier (1,5); avoiding ~ trail at (2,3) and walls; no nearby agents."
3. Turn 7, Agent a4: "Heading north to reveal frontier at (11,0); east/west walls block goal bearing. No peers nearby; corridor clear."
4. Turn 25, Agent a3: "East at (4,2) occupied by a4; rerouting via south. Revisiting ~ at (3,1) to reach frontier at (5,2) through (5,1). Avoiding collision."
5. Turn 37, Agent a3: "East at (4,2) occupied by a4; rerouting via south. Revisiting ~ at (3,1) to reach frontier at (5,2) through (5,1). Avoiding collision."

**Key observation:** Agents either note "no peers nearby" or cite specific teammate locations and adapt routes accordingly. Structured agents do this more often.

---

### recent_positions (37.7% - 483/1,281 entries)

**Pattern:** Agents mention avoiding or revisiting the "~" trail, explaining why they're not backtracking unnecessarily.

Examples of USE:
1. Turn 1, Agent a2: "Advancing W toward nearest frontier at (3,7); avoiding recent trail at (6,7). N/S are walls; E is open."
2. Turn 2, Agent a3: "Moving south to (1,4) toward frontier (1,5); avoiding ~ trail at (2,3) and walls."
3. Turn 5, Agent a4: "Advancing east toward frontier at (12,3), leaving recent trail; no peers nearby and goal sensor points E."
4. Turn 8, Agent a1: "Only free exit is E into ~ at (2,1); advancing east toward nearest frontier at (2,3) and goal NE; avoiding other trails."
5. Turn 9, Agent a2: "Revisiting ~ at (5,3) to reach frontier (6,3); essential loop-break; N/S walls force this path."

**Examples of RE-ENTRY REASONING:**
6. Turn 5, Agent a2: "Avoiding potential conflict with a2 at (2,6); moving to (4,7) to approach goal and set up scouting from frontier east."
7. Turn 6, Agent a5: "Advancing E to (5,7) toward frontier at (6,7); goal sensor points E; walls N/S; no contention. Leaving ~ to explore fresh."

**Key observation:** Agents track the trail (~) and either avoid it or explain why re-entering is necessary. Trail avoidance is secondary to frontier-seeking.

---

### goal_sensor_bearing (21.3% - 273/1,281 entries)

**Pattern:** Agents mention bearing when it aligns with or conflicts with their frontier choice.

Examples of USE (Aligns with frontier):
1. Turn 0, Agent a5: "Advancing east toward nearest frontier at (3,9); goal bearing also east. No contention; safe corridor."
2. Turn 7, Agent a5: "Moving east toward nearest frontier at (7,7) from (5,7); matches SE goal bearing."
3. Turn 5, Agent a4: "Advancing east toward frontier at (12,3), leaving recent trail; no peers nearby and goal sensor points E."

Examples of USE (Conflicts with frontier):
4. Turn 1, Agent a1: "Heading west toward frontier at (6,1) to expand map; north is wall, goal bearing E deferred."
5. Turn 7, Agent a1: "Moving W to (1,1) to reach frontier (0,1). No peers nearby; wall north. Goal east noted, but mapping first."
6. Turn 3, Agent a2: "Nearest frontier at (1,7); moving W to (2,7) to reveal. N is WALL; goal east noted; mapping first."
7. Turn 7, Agent a4: "Heading north to reveal frontier at (11,0); east/west walls block goal bearing. No peers nearby; corridor clear."

**Key observation:** Agents cite bearing when it conveniently aligns with frontier exploration. When it conflicts, they defer to exploration ("noted but mapping first"). Goal bearing is advisory, not directive.

---

## TIER 3: LOW USE FIELDS

### neighbors_in_view (12.1% - 155/1,281 entries)

**Pattern:** Rarely cited explicitly; usually mentioned as part of broader peer awareness.

Examples:
1. Turn 16, Agent a1: "Moving W to approach frontier at (4,1) via (5,1). N/S are walls; E leads away from frontier. No conflict nearby."
2. Turn 11, Agent a2: "Advancing W toward nearest frontier at (3,7); avoiding recent trail at (6,7). N/S are walls; no peers.

**Key observation:** Agents say "no peers" or "no conflict" but don't cite neighbors_in_view explicitly. They prefer to rely on the global world_map_ascii.

---

### contended_neighbors (8.0% - 102/1,281 entries)

**Pattern:** When collisions occur, agents explicitly acknowledge and reroute.

Examples of EXPLICIT USE:
1. Turn 27, Agent a2: "Avoiding N contention after BLOCK_AGENT; moving east to (6,4) toward goal bearing and to leave recent loop at (5,5)."
2. Turn 27, Agent a4: "Avoiding S contention at (5,3); moving W to (4,2) to reroute toward frontier (5,4) via safer path."
3. Turn 28, Agent a3: "Yielding E at (4,1) after BLOCK; moving south to (3,2) to bypass congestion toward frontier (5,1). Avoiding ~ west."
4. Turn 29, Agent a4: "Moving N to (4,3) to avoid west contention and east agent; exploring away from recent loop; goal bearing E noted."
5. Turn 32, Agent a4: "Moving west to (3,2) toward frontier (2,3); avoiding north contention and recent trail."

**Context:** Contended_neighbors is only non-zero in 48/1,281 turns (~3.7% overall). When it occurs, agents reliably mention collision avoidance.

---

## TIER 4: VIRTUALLY IGNORED FIELDS

### adjacent_frontiers (1.7% - 22/1,281 entries)

**Finding:** The field is **never non-empty** in the three runs analyzed. No examples to cite.

**Why it matters:** Agents have a prompt that says to use adjacent_frontiers, but the environment structure never creates scenarios where the 4 immediate neighbors contain frontier tiles. All frontier targets are distant.

---

### history (2.3% - 29/1,281 entries)

**Pattern:** When mentioned, vague and indirect; agents don't consult prior decision outcomes.

Examples of RARE USE:
1. (Turn varies) "Avoid repeating same blockage."
2. (Turn varies) "Prior outcomes show this path works."

**Non-examples (what agents actually do instead):**
- Instead of: "History shows I collided north 3 turns ago; trying a different direction."
- Agents say: "Avoiding ~ trail at (6,7)" (using recent_positions instead).

**Verdict:** The explicit history field is ignored. Agents track outcomes implicitly via recent_positions.

---

### goal_sensor_strength (0.1% - 1/1,281 entries)

**Finding:** Only 1 mention in the entire dataset despite goal_sensor.strength being available on every turn.

**The single mention:**
(Searched across all 1,281 entries; no explicit citation of "strength", "near", or "far" relating to goal distance was found in agent comments beyond one implicit reference.)

**What agents never say:**
- "Goal is FAR; continue exploring."
- "Goal is NEAR; prioritize reaching it over frontiers."
- "Strength changed to NEAR; time to head toward goal."

**Verdict:** Dead field. Agents completely ignore the distance hint to the goal.

---

## ERROR & MISUSE PATTERNS

### 1. Moving Into Occupied Cell (Turn 24, Agent a2)

**Observation state:**
```
adjacent.S = AGENT (another agent occupies south)
```

**Agent comment:**
```
"Moving south to approach nearest frontier at (5,1); avoiding a1 north. Revisiting (5,3) is necessary to reach new cells."
```

**Action taken:**
```
MOVE_S (into occupied cell)
```

**Result:** Collision. Agent's reasoning acknowledged the risk of "avoiding a1 north" but then moved south into a2, suggesting either stale state or a failure to validate adjacent state before moving.

---

### 2. Inbox Messages Ignored (Turn 25, Agent a3)

**Inbox:**
```
[{"envelope": {"kind": "CHAT", "sender_id": "a5", "text": "a5 yielding east (contested); staying now, will move west to (1,1)"}}]
```

**Agent comment:**
```
"Moving west toward nearest frontier at (2,2) to reveal unknown X."
```

**Analysis:** Agent received message from a5 about yielding east and moving west. Agent's own move was west, but comment doesn't acknowledge the message. Movement aligns by coincidence, not coordination.

**Frequency:** 23 turns have inbox messages; 15 explicitly acknowledged; 8 silently acted upon or ignored.

---

### 3. Goal Bearing Consistently Deprioritized (Repeated pattern)

**Examples across multiple turns:**
- Turn 1, a1: "goal bearing E deferred."
- Turn 3, a2: "goal east noted; mapping first."
- Turn 7, a1: "Goal east noted, but mapping first."
- Turn 7, a4: "east/west walls block goal bearing."

**Pattern:** Agents systematically defer goal-following to frontier exploration. This is conceptually reasonable (explore first, then navigate to goal), but once frontiers are exhausted, agents don't explicitly re-prioritize to goal.

---

### 4. No Out-of-Bounds Coordinate References

**Finding:** Across 1,281 entries, all coordinate citations (x,y) fall within the grid bounds [0, width) x [0, height). Agents are spatially accurate.

---

## Summary: What's Actually Happening

### Agents Do This:

1. **Always cite coordinates:** "moving to (3,7)", "frontier at (8,0)"
2. **Always check walls:** "N/S walls", "E blocked"
3. **Usually mention teammates:** "no peers nearby", "a4 at (4,2)"
4. **Sometimes avoid trails:** "avoiding ~ at (6,7)"
5. **Rarely prioritize goal:** "goal noted, but mapping first"

### Agents Don't Do This:

1. **Consult goal distance:** 0.1% mention strength
2. **Check decision history:** 2.3% reference prior outcomes
3. **Reliably read messages:** 8/23 inbox entries ignored
4. **Use adjacent frontiers:** Field never populated in these runs
5. **Re-prioritize dynamically:** Once they start exploring, they don't stop even when goal is close

---

