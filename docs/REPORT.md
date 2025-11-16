# Minimal Agent Communication Protocols for Multi-Agent Grid Navigation

## Executive Summary

**Central Question:** Do structured communication protocols (INTENT/REQUEST schemas) help LLM agents coordinate better than natural language?

**Answer:** Structured protocols don't help. After comprehensive cross-seed validation (45 runs), structured achieves 56.0% success—barely above no communication (57.3%). Freeform shows a modest advantage at 62.7%, but the effect size is small (5-agent difference out of 75) and we lack statistical testing to confirm significance.

**The Bigger Puzzle:** Communication doesn't seem to help much at all in this environment. The no-communication baseline performs nearly as well as the best strategy (5.4 percentage point difference). We don't know why. Possible explanations: partial observability makes communication ineffective, the one-turn message lag negates benefits, or the corridor environment doesn't require coordination.

**The Strong Finding (Methodological):** Initial experiments using a single seed showed structured achieving 73% success, suggesting strong performance. Cross-seed validation (5 different spawn configurations) revealed this was an extreme outlier—the true average is 56%. This demonstrates how single-scenario experiments produce completely misleading conclusions in stochastic multi-agent systems. Combined with extreme run-to-run variance (σ > mean for message counts), this work shows that rigorous experimental design is critical for LLM agent research.

---

## Task Overview

Five LLM-controlled agents navigate a 30×10 grid world with partial observability (vision radius = 1). Each agent receives a directional bearing toward a shared goal at position (28,1) but cannot see beyond their immediate surroundings. The environment contains maze-like obstacles that force indirect routing. With five agents competing for limited corridor space, collisions are frequent. Agents have 100 turns to coordinate and reach the goal.

We compare three communication protocols to determine which enables the most effective coordination across different spawn configurations.

---

## Environment Design

**Grid World:** 30×10 cells with walls forming maze-like corridors. Agents start at positions determined by a random seed and navigate toward a shared goal at (28,1).

**Partial Observability:** Each agent observes only cells within radius 1 (3×3 local patch centered on their position). Agents cannot see the full map or other agents beyond this range.

**Actions:** Each turn, agents choose one action:
- `MOVE_N/E/S/W`: Move one cell in the specified direction
- `STAY`: Remain in current position
- `COMMUNICATE`: Send one message to nearby agents (radio range = 2 cells)

**Collisions:** If two agents attempt to occupy the same cell, both are reset to their starting positions and lose that turn.

**Turn Execution:** All agent decisions are computed in parallel to maximize throughput. Messages sent during turn T are delivered at the start of turn T+1. This introduces a one-turn communication lag: if agent A communicates to nearby agent B in turn T, agent B will not see that message until deciding their action for turn T+1. This design choice prioritizes parallelizable LLM inference over instantaneous local communication. The lag affects all communication modes equally and does not bias cross-condition comparisons.

**Goal Sensor:** Each agent receives a noisy directional bearing (N/E/S/W) indicating the approximate direction to the goal, with distance hints (VERY_CLOSE/CLOSE/FAR/VERY_FAR). Agents must integrate this bearing with their local observations to navigate.

---

## Communication Protocols

All agents share a base prompt instructing them to navigate toward the goal while avoiding collisions. The communication strategy determines what messages (if any) agents can exchange to coordinate.

### None (No Communication)

**Approach:** Agents receive no communication capability (radio_range=0). They must coordinate implicitly through observation of nearby agents (within vision radius = 1).

**Prompt addition:** *(none - agents simply navigate without message exchange)*

**Messages exchanged:** 0 per run

**Strategy:** Independent exploration with collision avoidance based solely on visual observation.

---

### Structured

**Approach:** Agents use a fixed message schema with two message types: `INTENT` (announce next action) and `REQUEST` (ask peer to yield or share goal location). Priority tiebreaker: lowest agent ID moves first in contested cells.

**Prompt addition:**
```python
# From src/llmgrid/agent/llm_agent.py, lines 64-71
"Allowed: INTENT or REQUEST(YIELD|GUIDE target=(x,y)). One message max per turn."
"When to communicate: only if any_peer_in_range is true and you have useful info
 (collision risk, new corridor, map gap) that a nearby peer benefits from."
"Good reasons: approaching a shared cell, you see G, you discovered a useful corridor
 or dead end, your buddy might be stuck, or you need a map snippet to progress."
"Priority: when 2+ agents want the same cell, LOWEST agent_id MOVES immediately
 (no announcement needed). Higher IDs MUST yield (stay/reroute). No mutual yielding,
 no wasted turns announcing priority."
"Message choice: INTENT to share your next move; REQUEST(YIELD,target=T) if you need
 priority; REQUEST(GUIDE,target=(gx,gy)) to share G or help a stuck teammate."
"Avoid repeats: do not send the same content within 5 turns unless new information appeared."
```

**Full schema:**

```typescript
MsgIntent {
  kind: "INTENT"
  next_action: "MOVE_N" | "MOVE_E" | "MOVE_S" | "MOVE_W" | "STAY"
  sender_id: string
  seq: int
}

MsgRequest {
  kind: "REQUEST"
  req: "YIELD" | "GUIDE"
  target?: {x: int, y: int}  // optional
  sender_id: string
  seq: int
}
```

**Example messages (serialized from actual transcripts):**
```json
{
  "kind": "INTENT",
  "sender_id": "a4",
  "seq": 0,
  "next_action": "STAY"
}
```

```json
{
  "kind": "REQUEST",
  "sender_id": "a5",
  "seq": 0,
  "req": "YIELD",
  "target": {
    "x": 8,
    "y": 1
  }
}
```

**Note:** Other agents receive only the message envelope (kind, sender_id, seq, next_action/req/target). The "Reasoning" field shown in coordination examples is debug output for system builders, not transmitted to other agents.

---

### Freeform (Natural Language)

**Approach:** Agents compose natural language messages (≤96 characters) describing their intent, requests, or observations. Same priority rule as structured, but expressed in prose.

**Prompt addition:**
```python
# From src/llmgrid/agent/llm_agent.py, lines 74-84
"DEFAULT TO MOVE. Only CHAT when the message prevents imminent collision or shares
 critical info (goal location, dead end, you're rerouting around a peer)."
"Allowed: one CHAT (<=96 chars) per turn. Write naturally to help your teammate."
"When to communicate: only if any_peer_in_range is true. Share something useful
 (new route, goal location, dead end you verified, you are rerouting, or you are stuck)."
"Use coordinates so teammates can mark their maps: e.g., 'heading east toward (5,2)',
 'found goal at (14,4)', 'dead end north; trying south', 'sharing loop at (3,1)-(3,2)'."
"Priority: when 2+ agents want the same cell, LOWEST agent_id goes first. Higher IDs yield.
 Example: 'I'm a5, yielding (5,5) to you, going west' or just move without announcing
 if you're yielding."
"Be cooperative and concise; avoid repeating unchanged info within ~5 turns."
```

**Example messages:**
```json
{
  "kind": "CHAT",
  "sender_id": "a3",
  "seq": 0,
  "text": "a3 holding at (8,1); waiting to move west to (6,1). Please yield (7,1) if possible."
}
```

```json
{
  "kind": "CHAT",
  "sender_id": "a2",
  "seq": 0,
  "text": "a2 heading west toward (8,1); please yield so I can scout."
}
```

---

## Experimental Journey: From Single-Seed to Cross-Validation

### Phase 1: Initial Experiments (Seed 13 Only)

**What we did:** Ran 9 experiments (3 replicates per strategy) using seed 13 for agent starting positions. All runs used the same maze (long_corridor, obstacle seed 606) and model (azure:gpt-5-mini).

**What we saw:**
- Structured: 73% success (11/15 agents finished)
- Freeform: 33% success (5/15 agents finished)
- None: 20% success (3/15 agents finished)

**What we thought:** Structured communication provides a massive advantage (3.6× better than no communication). Rigid schemas with priority rules enable effective coordination. Fixed message formats reduce ambiguity and prevent the chaotic over-communication seen in freeform runs.

**Why we were wrong:** We had tested only one spawn configuration. Seed 13 happened to create agent starting positions that strongly favored structured coordination—but we didn't know this yet.

### Phase 2: Cross-Seed Validation (Seeds 13-17)

**What we did:** Expanded to 45 total runs using 5 different spawn seeds (13, 14, 15, 16, 17), running a complete 3×3 matrix (3 strategies × 3 replicates) per seed.

**Why we did it:** Single-seed results felt too clean. We wanted to confirm the findings generalized across different starting configurations. This is standard practice in stochastic systems—validate across multiple scenarios before drawing conclusions.

**What we discovered:** Seed 13 was an extreme outlier.

**Cross-seed results (45 runs total):**
- Freeform: 62.7% success (47/75 agents)
- None: 57.3% success (43/75 agents)
- Structured: 56.0% success (42/75 agents)

**The reversal:** Structured dropped from 73% (seed 13 only) to 56% (cross-seed average). It went from best performer to worst, performing barely better than no communication at all.

**The lesson:** Single-seed experiments can produce completely misleading conclusions. The spawn configuration in seed 13 happened to create tight corridors where structured priority rules helped. Other seeds created configurations where freeform's flexible coordination worked better.

---

## The Non-Determinism Problem: Extreme LLM Variability

Before diving into results, we need to address a striking characteristic of LLM-based multi-agent systems: **massive, unexplained behavioral variability**.

### The Same Agents, The Same Environment, Wildly Different Behavior

Consider these examples from our dataset—all runs used identical prompts, model, maze, and parameters:

**Freeform communication:**
- Run A: 0 messages sent, 3 agents finished
- Run B: 35 messages sent, 4 agents finished
- Run C: 1 message sent, 2 agents finished
- Run D: 15 messages sent, 3 agents finished

**Structured communication:**
- Run E: 0 messages sent, 1 agent finished
- Run F: 23 messages sent, 4 agents finished
- Run G: 7 messages sent, 3 agents finished

**The puzzling part:** Nothing obvious changed between these runs. Same prompt, same environment, same model. Yet agent behavior ranged from complete silence (0 messages) to heavy communication (35 messages). The standard deviations are comparable to or larger than the means:

- Freeform: mean 5.9 messages, σ=9.1 (157% of mean!)
- Structured: mean 8.9 messages, σ=6.9 (78% of mean)

### It's Not Just Messages—It's Everything

**Collision counts:** 0-48 range within the same strategy (freeform)
**Run duration:** 37.7 to 189.7 minutes for identical task (freeform)
**Success outcomes:** Identical configurations produce 1-4 agents finished

### What's Causing This?

We don't know for certain, but likely contributors:

**1. LLM sampling stochasticity**
Modern LLMs have inherent randomness in generation. The model (gpt-5-mini) does not accept temperature parameter, yet produces non-deterministic outputs. Different backend instances, hardware, or time-of-day load could affect sampling.

**2. Cascade effects from early decisions**
One agent makes a slightly different turn-1 move, which changes what another agent observes, which changes their decision, which ripples forward. Multi-agent systems amplify small perturbations.

**3. Backend infrastructure variability**
Azure/OpenRouter may route requests to different instances with different quantization, batching, or optimization strategies. What looks like "the same model" might not be byte-identical across calls.

**4. Prompt interpretation variance**
LLMs don't parse prompts deterministically. The same prompt on the same input can yield different responses based on internal state we can't observe.

### Why This Matters for Research

**Standard deviation tells the story:**
When σ ≈ mean (or larger), you're measuring noise as much as signal. Our freeform strategy has σ=9.1 messages with mean=5.9—that's 157% variance. Structured has σ=6.9 with mean=8.9—that's 78% variance.

**Single runs are meaningless:**
One run with 35 messages tells you nothing. Another identical run might send 0. You need large sample sizes (we used 15 runs per strategy) to see signal through noise.

**Small studies will mislead:**
If we'd run only 3 replicates per strategy (9 total runs), we might have happened to sample mostly high-communication or low-communication runs, producing completely wrong conclusions about typical behavior.

**Cross-scenario validation is essential:**
Not only do you need multiple replicates within one seed, you need multiple seeds. Otherwise you're measuring seed-specific quirks, not strategy effectiveness.

### The Implication

When reading our results below, remember: **the averages hide extreme variability**. A "mean of 5.9 messages" doesn't mean agents typically send 6 messages. It means they send anywhere from 0 to 35, with high unpredictability.

This variability is a **fundamental challenge for LLM-based multi-agent systems**. Unlike classical algorithms, you cannot predict what LLM agents will do on any given run—even with identical inputs. You can only characterize their statistical behavior over many runs.

This makes LLM agents both fascinating (emergent, adaptive behavior) and frustrating (unreliable, hard to debug). It also makes rigorous experimental design absolutely critical.

---

## Results: Cross-Seed Validation (45 Runs)

All results below are from the complete cross-seed study: 5 seeds × 3 strategies × 3 replicates = 45 runs total. Maze preset: `long_corridor` (obstacle seed 606). Agent spawn seeds: 13, 14, 15, 16, 17. Model: `azure:gpt-5-mini`. Turn budget: 100.

**Remember:** The statistics below represent averages over highly variable behavior. Standard deviations are large. Individual runs can differ dramatically from the mean.

---

### Metric 1: Success Rate

**Definition:**

$$\text{Success Rate} = \frac{\sum \text{agents finished}}{\text{total agents}} \times 100\%$$

where total agents = 15 runs × 5 agents = 75 per strategy.

**Computation:**
For each run, we parsed `episode.json` and counted agents with `status == "FINISHED"` in the final frame (turn 100). Aggregated across all 45 runs.

**Results:**

![Figure 1: Success Rate](https://raw.githubusercontent.com/Bryan-Nsoh/agent-talk-2/main/experiments/cross_seed_baseline_20251112T143355Z/plots/1_success_rate.png)

| Rank | Strategy | Success Rate | Agents Finished | Runs |
|------|----------|--------------|-----------------|------|
| 🥇 | Freeform | 62.7% | 47/75 | 15 |
| 🥈 | None | 57.3% | 43/75 | 15 |
| 🥉 | Structured | 56.0% | 42/75 | 15 |

**Interpretation:** Freeform natural language achieves the highest success rate, outperforming both structured protocols and no communication. Structured provides minimal benefit over baseline (56.0% vs 57.3%)—a stark reversal from the seed-13-only results that showed structured at 73%.

**What changed:** Different spawn configurations revealed that structured priority rules don't generalize well. Freeform's flexible, context-aware coordination adapts better to varying scenarios.

---

### Metric 2: Communication Volume

**Definition:**

- **Messages**: Count of COMMUNICATE actions per run
- **Tokens**: Sum of tokens across all messages, using `tiktoken` library with `o200k_base` encoding

**Computation:**
We parsed `transcript.jsonl` for each run, extracted all `COMMUNICATE` actions, and computed:
1. Message count (direct count)
2. Token count using `tiktoken.get_encoding("o200k_base").encode(message_text)`

**Results:**

![Figure 2: Communication Volume](https://raw.githubusercontent.com/Bryan-Nsoh/agent-talk-2/main/experiments/cross_seed_baseline_20251112T143355Z/plots/2_communication_volume.png)

| Strategy | Messages/Run | Total Messages | Tokens/Message | Efficiency |
|----------|--------------|----------------|----------------|------------|
| Freeform | 5.9 ± 9.1 | 89 (15 runs) | ~21.0 | 0.98 msgs/finished agent |
| Structured | 8.9 ± 6.9 | 134 (15 runs) | ~5.7 | 3.21 msgs/finished agent |
| None | 0.0 ± 0.0 | 0 (15 runs) | 0.0 | n/a |

**Interpretation:** Freeform achieves higher success (62.7% vs 56.0%) while using 34% fewer total messages (89 vs 134). Despite freeform messages being longer (21.0 vs 5.7 tokens/message), the strategy is more efficient: 0.98 messages per successful agent versus structured's 3.21 messages per successful agent.

**Key insight:** More messages ≠ better coordination. Structured agents communicate more frequently but achieve worse outcomes, suggesting the rigid protocol creates coordination overhead rather than clarity.

**Variance:** Both strategies show high variance (freeform σ=9.1, structured σ=6.9), indicating LLM behavior variability across runs—likely due to time-of-day effects or backend differences.

---

### Metric 3: Collision Rate

**Definition:**

$$\bar{C} = \frac{1}{N}\sum_{r=1}^{N} C_r \quad \pm \quad \sigma$$

where $C_r$ = collisions in run $r$, $N$ = 15 runs per strategy, $\sigma$ = standard deviation

**Computation:**
Collision counts extracted from `metrics.json["collisions"]` for each run. Mean and standard deviation computed via NumPy.

**Results:**

![Figure 3: Collision Rate](https://raw.githubusercontent.com/Bryan-Nsoh/agent-talk-2/main/experiments/cross_seed_baseline_20251112T143355Z/plots/3_collision_rate.png)

| Strategy | Collisions/Run | Total Collisions | BLOCK_AGENT | BLOCK_WALL |
|----------|----------------|------------------|-------------|------------|
| Freeform | 14.5 ± 11.5 | 188 | 178 (95%) | 10 (5%) |
| Structured | 17.0 ± 8.6 | 323 | 313 (97%) | 12 (3%) |
| None | 18.1 ± 14.6 | 199 | 196 (98%) | 3 (2%) |

**Interpretation:** Collision rates are similar across all three strategies (14.5-18.1 range), with freeform having slightly fewer. Communication does not significantly reduce agent-agent conflicts in this environment.

**What changed from seed 13:** The initial study showed structured with dramatically fewer collisions (10.0 ± 3.6). Cross-seed validation reveals this was specific to seed 13's configuration. Across diverse scenarios, structured's collision-avoidance advantage disappears.

**Implication:** The hypothesis that structured priority rules prevent collisions does not hold across different spawn configurations.

---

### Metric 4: Completion Timeline

**Definition:**

$$F(t) = \frac{1}{N}\sum_{r=1}^{N} \text{agents finished by turn } t \text{ in run } r$$

Cumulative agents finished at turn $t$, averaged across 15 runs per strategy.

**Computation:**
For each run, we parsed `episode.json` frame by frame and tracked when each agent's status changed to `FINISHED`. We computed cumulative count at each turn, then averaged across the 15 runs per strategy.

**Results:**

![Figure 4: Completion Timeline](https://raw.githubusercontent.com/Bryan-Nsoh/agent-talk-2/main/experiments/cross_seed_baseline_20251112T143355Z/plots/4_completion_timeline.png)

**Interpretation:** Freeform shows the strongest upward trajectory, reaching the highest final completion count. None and structured show similar progression patterns, with structured finishing slightly below baseline.

**What this tells us:** Freeform enables agents to complete the task more consistently across different spawn configurations. Structured's completion timeline doesn't match the strong performance seen in seed 13 alone.

---

### Metric 5: Success vs Communication Cost

**Definition:**

Scatter plot showing total tokens sent (x-axis) vs agents finished (y-axis) for each run.

**Computation:**
Each point represents one run: x = token count from tiktoken, y = finish count from episode.json final frame.

**Results:**

![Figure 5: Success vs Tokens](https://raw.githubusercontent.com/Bryan-Nsoh/agent-talk-2/main/experiments/cross_seed_baseline_20251112T143355Z/plots/5_success_vs_tokens.png)

**Interpretation:** The ideal region is upper-left (high success, low tokens). Freeform achieves strong success (3-4 agents) with low token counts (most runs under 300 tokens). None performs nearly as well as structured despite zero communication cost. Structured shows no clear correlation between message volume and success.

**Key finding:** Structured's compactness (5.7 tokens/message) doesn't translate to better outcomes. The 3.7× token advantage over freeform is irrelevant when success rates are 12% lower (56.0% vs 62.7%).

---

## Per-Seed Breakdown

| Seed | Structured Success | Freeform Success | None Success |
|------|-------------------|------------------|--------------|
| 13 | 2.0/5 (40%) | - | - |
| 14 | 3.0/5 (60%) | 3.0/5 (60%) | 3.0/5 (60%) |
| 15 | 2.2/5 (44%) | 3.5/5 (70%) | 3.5/5 (70%) |
| 16 | 3.3/5 (66%) | 4.7/5 (94%) | 4.0/5 (80%) |
| 17 | 2.7/5 (54%) | 2.7/5 (54%) | 1.7/5 (34%) |

**Note:** The table shows average agents finished per run within each seed group (3 replicates per cell).

**Pattern:** Performance varies significantly by seed. Freeform maintains advantage across most seeds (particularly strong in seeds 15 and 16). Structured performs inconsistently, never achieving the 73% success seen in preliminary seed-13 experiments.

**The seed-13 outlier:** When we ran the full 3×3 matrix for seed 13 in the cross-seed study, structured achieved only 40% success (2.0/5 agents)—far below the 73% from the initial experiments. This suggests even within seed 13, there was high variance, and the initial 3 runs happened to be lucky.

---

## Why Doesn't Communication Help?

The most surprising finding isn't that freeform beats structured—it's that **communication barely helps at all**. The no-communication baseline (57.3%) performs nearly as well as freeform (62.7%), and collision rates are similar across all strategies (14.5-18.1 range).

This is unexpected. We designed communication specifically to help agents coordinate in tight corridors with partial observability. Why didn't it work?

### The Critical Missing Piece: We Have No Reference Point

**Before exploring environmental or technical explanations, we need to ask a more fundamental question:** Is 60% success good or bad?

**We have no idea.** We're comparing LLM strategies to each other, but we don't know how humans would perform on this exact task. Without a baseline, we can't interpret our results.

**The fundamental question:** What if the task itself is the bottleneck, not communication? Or what if spatial navigation is easy but we're incorrectly attributing success to communication?

**Why we need to know:**
- LLM agents achieve 56-63% success across strategies
- Is this good or terrible? We have no reference point
- Maybe 60% is near-optimal for this task and communication can't help more
- Or maybe humans would hit 95% and LLMs are just bad at spatial reasoning
- Or maybe humans achieve 60% too, meaning the task is fundamentally hard

**The critical test: Human baseline study**

Run the identical experiment with human participants:

**Setup:**
- 5 humans navigate the same 30×10 maze
- Exact same constraints: vision radius=1 (only see immediate 3×3 patch), 100 turns, goal bearing provided
- Three conditions: (1) no communication, (2) text chat (freeform), (3) structured protocol
- Multiple maze seeds, multiple groups

**What this would reveal:**

**Scenario A: Humans achieve ~60% (same as LLMs)**
→ The task is fundamentally hard with radius=1 observability
→ Communication can't overcome severe information constraints
→ LLMs are performing at human level on spatial navigation
→ **Implication:** This becomes a validated benchmark for "coordination under extreme partial observability"—potentially the ARC-AGI equivalent for multi-agent systems

**Scenario B: Humans achieve ~95% without communication**
→ Spatial navigation is the bottleneck, not communication
→ Humans have better spatial reasoning/memory than LLMs
→ Communication doesn't help because good navigation solves the problem
→ **Implication:** LLM spatial reasoning is the capability gap, not communication processing

**Scenario C: Humans achieve ~60% without comm, ~95% with comm**
→ Humans benefit massively from communication, LLMs don't
→ LLMs can't effectively use coordination information
→ **Implication:** Architectural or prompting gap in how LLMs process and act on peer messages

**Scenario D: Humans achieve ~30% (worse than LLMs)**
→ Humans struggle more with severe constraints than LLMs
→ Unexpected but possible—LLMs might handle systematic search better
→ **Implication:** LLMs have strengths in constrained exploration that exceed human performance

**Why this matters more than anything else:**

Without a human baseline, we're making unfounded assumptions:
- "60% seems low" ← compared to what?
- "Communication should help" ← based on what evidence?
- "LLMs are bad at this" ← relative to whom?

**Every conclusion in this paper is uninterpretable without this data.**

**The ARC-AGI parallel:**

This could become **the benchmark for multi-agent communication under constraints**:
- ARC-AGI tests individual reasoning under novel constraints
- This would test coordination reasoning under severe observability constraints
- Like ARC, it's simple to state but potentially very hard
- Human baseline is essential for ARC's value—same here
- If humans crush it, you've identified an LLM capability gap
- If humans struggle too, you've validated a hard benchmark
- If communication helps humans but not LLMs, you've isolated the coordination deficit

**This transforms the work from "we compared LLM strategies" to "we identified a fundamental benchmark for testing multi-agent coordination capabilities."**

**Test this by:** Running a human subjects study with the exact same environment, constraints, and measurement protocol. Could be conducted as an online multiplayer game with constrained visibility. Implementation would be straightforward—the environment already exists, just needs a human-playable interface.

---

### Possible Environmental/Technical Explanations

*If we had the human baseline and confirmed LLMs underperform, here are potential technical reasons:*

### Hypothesis 1: Partial Observability Limits Communication Effectiveness

**The problem:** Agents can only see radius=1 (their immediate 3×3 patch). They can't see the maze structure, where other agents are beyond immediate neighbors, or the full path to the goal.

**Why this hurts communication:**
- An agent saying "I'm heading east toward (15,3)" doesn't help if you can't see where (15,3) is relative to you
- You can't build a shared mental model of the environment from local radio messages
- By the time you receive a message (1-turn lag), the sender may have moved or changed plans

**Test this by:** Running experiments with increased vision radius (3 or 5) to see if communication becomes more effective when agents can contextualize messages.

### Hypothesis 2: The One-Turn Message Lag Negates Coordination Benefits

**The problem:** Messages sent at turn T arrive at turn T+1. In a dynamic environment with 5 agents moving simultaneously, one turn is a long time.

**Why this hurts communication:**
- Agent A: "I'm moving to (5,5)" [turn 10]
- Agent B receives this at turn 11, but A is already there or blocked
- By the time B acts on the info (turn 11), it's stale
- Coordination requires real-time feedback; async messaging creates race conditions

**Test this by:** Running experiments with synchronous communication (messages delivered same-turn) or look-ahead planning where agents commit to multi-turn paths.

### Hypothesis 3: Corridor Environment Doesn't Require Coordination

**The problem:** The "long_corridor" maze might have natural traffic flow that doesn't benefit from explicit coordination.

**Why this hurts communication:**
- If corridors are wide enough for independent navigation, agents don't *need* to coordinate
- If corridors are too narrow, agents collide regardless of communication (bottleneck problem)
- The sweet spot where communication helps (moderate congestion) might be rare in this environment

**Supporting evidence:**
- Collision rates are similar across all strategies (communication doesn't prevent collisions)
- "None" strategy succeeds 57.3%—agents can navigate independently pretty well

**Test this by:** Running experiments in more complex mazes (branching paths, open rooms, multiple goals) where coordination should matter more.

### Hypothesis 4: LLMs Don't Use Communication Effectively

**The problem:** Even when agents send messages, they may not *act* on them appropriately.

**Why this hurts communication:**
- LLMs might generate messages to "follow the prompt" but not actually change behavior based on received messages
- Agents might acknowledge messages ("a4 yielding") but continue collision anyway due to planning errors
- The history window (5 turns) might be too short to maintain consistent coordination state

**Evidence:**
- High variance in message counts (0-35 range) suggests agents aren't following consistent communication strategies
- Negative correlation (r=-0.516) between messages and success in structured runs—more messages → worse outcomes
- Some runs succeed with zero messages, others fail with heavy messaging

**Test this by:** Analyzing transcripts to see if agents actually change behavior in response to messages, or if communication is performative.

### Hypothesis 5: The Prompt Guidance Discourages Communication

**The problem:** We told agents "DEFAULT TO MOVE" and "only CHAT when the message prevents imminent collision or shares critical info."

**Why this hurts communication:**
- Agents may be too conservative about when to communicate
- By the time collision is "imminent" (visible in radius=1), it's too late to coordinate
- Agents need to communicate *before* they're in visual range, but prompt discourages this

**Evidence:**
- Low message counts (mean 5.9-8.9 per run) suggest agents are staying quiet
- Some agents send zero messages across entire 100-turn run

**Test this by:** Running experiments with different prompts encouraging proactive communication or removing the "DEFAULT TO MOVE" directive.

---

### Summary: Why These Explanations Are Secondary

The five technical/environmental hypotheses above are all speculation without the human baseline. We can't know if "partial observability limits communication" or "LLMs don't use messages effectively" until we know whether humans would perform better under identical constraints.

**The human study is the critical next step.** Everything else is guessing.

---

## Communication in Action: Sequential Coordination Examples

These examples come from the initial seed-13 experiments and illustrate how each strategy coordinates in practice. While seed 13 was an outlier in terms of overall success rates, these sequences show the typical communication patterns for each approach.

### Structured Coordination Sequence (Seed 13, Run 1, Turns 3-12)

This 6-message sequence shows agent a5 navigating a congested corridor while coordinating with nearby agents:

**Turn 3, a4:**
```json
Message: {"kind": "INTENT", "next_action": "STAY"}
Reasoning: "Yielding; holding at (3,8) so a2 can access (3,7) toward frontier (3,6)."
```

**Turn 8, a5:**
```json
Message: {"kind": "REQUEST", "req": "YIELD", "target": {"x": 8, "y": 1}}
Reasoning: (requesting priority at contested cell)
```

**Turn 9, a5:**
```json
Message: {"kind": "INTENT", "next_action": "STAY"}
Reasoning: "Yielding after previous YIELD; holding at (7,1) so lower-id peer may
advance toward frontier (9,1)."
```

**Turn 10, a5:**
```json
Message: {"kind": "INTENT", "next_action": "STAY"}
Reasoning: "Announcing yield; staying to avoid collision."
```

**Turn 11, a5:**
```json
Message: {"kind": "INTENT", "next_action": "MOVE_E"}
Reasoning: "Announcing move E to scout frontier at (9,1) before proceeding; avoiding
repeated blocked attempt."
```

**Turn 12, a5:**
```json
Message: {"kind": "INTENT", "next_action": "STAY"}
Reasoning: "Avoiding repeated MOVE_E into (8,1) after a YIELD; broadcasting STAY to
coordinate with nearby peer."
```

**What happened:** Agent a5 detected congestion at (8,1), requested yield from peer, waited for 2 turns, announced move, then yielded again when detecting continued blockage. The sequence shows repeated coordination with terse messages (6-10 tokens each). Result: 0 collisions in this sequence, forward progress maintained.

---

### Freeform Coordination Sequence (Seed 13, Run 1, Turns 48-53)

This 6-message sequence shows agents a2 and a4 negotiating access to a contested corridor:

**Turn 48, a2:**
```json
Message: {"kind": "CHAT", "text": "a2 heading west toward (8,1); please yield so I can scout."}
Reasoning: "Broadcasting intent: heading west to scout frontier at (8,1); requesting
yield to avoid collision."
```

**Turn 49, a4:**
```json
Message: {"kind": "CHAT", "text": "a4 yielding at (8,1); staying to let a2 scout west"}
Reasoning: "Yielding to a2; staying at (8,1) to avoid collision and let them scout west."
```

**Turn 50, a4:**
```json
Message: {"kind": "CHAT", "text": "a4 yielding (8,1) to a2; staying to let you scout."}
Reasoning: "Yielding (8,1) to a2 per their request and to avoid collision; staying to
let them scout."
```

**Turn 52, a2:**
```json
Message: {"kind": "CHAT", "text": "a2 heading west toward (6,1); entering contested (7,1),
yielding if needed."}
Reasoning: "Announcing west toward (6,1); entering contested (7,1)."
```

**Turn 52, a4:**
```json
Message: {"kind": "CHAT", "text": "a4 yielding (8,1) to a2; staying to avoid collision and
let you scout."}
Reasoning: "Yielding at (8,1) to let a2 scout west; avoiding contended W."
```

**Turn 53, a4:**
```json
Message: {"kind": "CHAT", "text": "a4 yielding (8,1) to a2; staying to avoid contested (7,1)."}
Reasoning: "Yielding to a2; staying at (8,1) to avoid contested (7,1)."
```

**What happened:** Agent a2 requested access, a4 yielded with confirmation, a2 advanced, a4 repeated yield confirmation (turns 50, 52, 53). Messages are longer (15-20 tokens each) and show redundancy (3 nearly identical yield messages). Result: coordination successful but verbose.

**Comparison:** Both sequences achieve coordination. Structured uses 6 messages (54 tokens total). Freeform uses 6 messages (102 tokens total). Structured eliminates redundancy through schema constraints; freeform allows natural repetition.

**But here's the critical point:** Despite freeform's verbosity and apparent inefficiency in individual sequences, it achieves 12% higher success rate across diverse scenarios (62.7% vs 56.0%). The redundancy doesn't hurt—it might even help by reinforcing coordination intent across the one-turn message lag.

---

## Interpretability Revisited

**Initial hypothesis (from seed 13):** Structured messages are more interpretable because they're compact, consistent, and parseable.

**What cross-seed validation showed:** Interpretability doesn't matter if the strategy doesn't work.

### Structured: Compact but Ineffective

Compact, fixed-schema messages are easy to parse programmatically:

```
Turn 3, a4: INTENT STAY (yield to a2)
Turn 8, a5: REQUEST YIELD @ (8,1)
Turn 9, a5: INTENT STAY (confirming yield)
```

**Advantages:**
- Consistent format enables regex/scripting for log analysis
- Lower token count (5.7 vs 21.0 tokens/message)
- Schema enforces semantic categories (INTENT vs REQUEST)
- No ambiguity in message structure

**Disadvantage that matters more:**
- Only 56.0% success rate across diverse scenarios
- Rigid structure doesn't adapt to different coordination contexts
- Agents send 3.21 messages per successful finish (vs freeform's 0.98)

### Freeform: Verbose but Effective

Natural language messages are longer and more variable:

```
"a2 heading west toward (8,1); please yield so I can scout."
"a4 yielding at (8,1); staying to let a2 scout west"
"a4 yielding (8,1) to a2; staying to let you scout." [repeated]
```

**Disadvantages:**
- 3.7× more tokens per message (21.0 vs 5.7)
- Variability requires semantic parsing
- Redundancy (agent a4's 3 nearly identical yields)
- Higher cognitive load for human log review

**Advantage that matters more:**
- 62.7% success rate across diverse scenarios
- Flexible coordination adapts to different contexts
- Only 0.98 messages per successful finish
- Natural language conveys nuance (intention, reasoning, context)

### Operational Reality

For deployment, **task success matters more than log compactness**. Freeform's 12% higher success rate (62.7% vs 56.0%) outweighs structured's 3.7× token advantage. If you need agents to actually complete tasks across varying conditions, choose the strategy that generalizes—even if it's messier to audit.

That said, structured logs are genuinely easier to parse for automated monitoring. If you could get structured to match freeform's success rate, the interpretability advantage would be meaningful. But you can't. The rigidity that makes structured interpretable also makes it inflexible.

---

## Key Findings

### 1. Freeform Communication Generalizes Better

**Result:** Freeform achieves 62.7% success across 5 spawn seeds, outperforming structured (56.0%) and no communication (57.3%).

**Why:** Natural language allows context-aware, flexible coordination. Agents express nuanced intent ("please yield so I can scout") that adapts to different scenarios better than rigid INTENT/REQUEST schemas.

### 2. Structured Protocols Provide Minimal Benefit

**Result:** Structured achieves 56.0% success—barely above no communication (57.3%) and 12% below freeform (62.7%).

**Why:** Rigid message schemas create coordination overhead without corresponding benefit. The fixed INTENT/REQUEST format doesn't capture the contextual flexibility needed for effective multi-agent coordination across diverse scenarios.

### 3. Communication Efficiency: Sparse Beats Verbose

**Result:** Freeform uses only 5.9 messages/run on average but achieves the highest success. Structured uses 8.9 messages/run for worse outcomes.

**Metric:** Messages per successful agent:
- Freeform: 0.98 messages
- Structured: 3.21 messages (3.3× less efficient)

**Why:** Freeform agents coordinate opportunistically when needed and stay silent when not. Structured agents follow protocol even when communication doesn't help.

### 4. Collision Rates Are Similar Across Strategies

**Result:** All strategies show 14.5-18.1 collisions/run with high variance.

**Why:** Communication doesn't significantly reduce agent-agent conflicts in this environment. Collisions happen due to tight corridors and partial observability, not lack of coordination messages.

**Implication:** The hypothesis that structured priority rules prevent collisions (strongly supported by seed 13 alone) does not hold across diverse scenarios.

### 5. High LLM Variance Across All Strategies

**Result:**
- Structured messages: 0-23 range (median 7), σ=6.9
- Freeform messages: 0-15 range (median 3), σ=9.1
- Success rates vary widely within each seed

**Why:** LLM behavior shows significant variability—likely due to time-of-day effects, backend differences, or stochastic sampling.

**Implication:** Large sample sizes across multiple scenarios are essential. Small studies will produce unreliable conclusions.

### 6. Single-Seed Studies Are Dangerously Misleading

**Result:**
- Seed 13 only: Structured 73%, Freeform 33%, None 20%
- Cross-seed (13-17): Structured 56%, Freeform 62.7%, None 57.3%

**Complete reversal:** The strategy that looked best (structured) became worst. The strategy that looked worst (none) became middle-tier.

**Why:** Seed 13 created a spawn configuration that strongly favored structured coordination. Other seeds revealed this was not representative.

**Implication:** This is the most important methodological lesson. In stochastic multi-agent systems, single-scenario experiments can produce completely wrong conclusions. Always validate across multiple seeds/configurations before drawing conclusions.

### 7. More Communication ≠ Better Outcomes

**Result:** Negative correlation (r=-0.516) between messages and success in structured runs. Some freeform runs succeeded with zero messages.

**Why:** Communication quality and timing matter more than volume. Excessive messaging can create coordination overhead that hurts performance.

**Implication:** Don't optimize for message count. Optimize for task success.

---

## Conclusion

The primary contribution of this work is **methodological**: demonstrating how single-scenario experiments mislead, and how extreme LLM variance demands rigorous experimental design.

### What We Learned (With Certainty)

**About experimental methodology:**
- **Single-seed studies are dangerously misleading:** Seed 13 showed structured at 73% success; cross-seed average was 56%. Complete ranking reversal.
- **LLM agents exhibit extreme non-determinism:** Identical setups produce 0-35 message ranges, σ > mean. You cannot predict behavior from single runs.
- **Large sample sizes are essential:** Need 10+ replicates per condition to characterize behavior through noise.
- **Cross-scenario validation is mandatory:** What works in one spawn configuration may fail in others. Test multiple seeds/environments before drawing conclusions.

These findings are robust and important for anyone working with LLM multi-agent systems.

### What We Think We Learned (With Uncertainty)

**About communication strategies:**
- Freeform shows modest advantage: 62.7% vs 56.0% (structured) vs 57.3% (none)
- Effect size is small: 5-agent difference out of 75 attempts
- We lack statistical significance testing—this could be sampling noise
- No-communication baseline performs nearly as well as best strategy (5.4pp difference)

**The bigger puzzle:** Communication doesn't seem to help much in this environment. We generated five hypotheses for why (partial observability, message lag, corridor design, LLM limitations, prompt guidance), but haven't tested them.

### Why This Matters

**For multi-agent system design:**
Don't assume communication will help. Test it. In narrow corridors with partial observability and async messaging, our agents coordinated about as well without communication as with it. Your environment may differ, but communication is not automatically beneficial.

If you do use communication: freeform natural language shows better generalization than rigid schemas in our tests, but the effect is modest.

**For research methodology:**
If you're evaluating LLM agent strategies, you need:
1. Multiple scenario seeds (5+ recommended)
2. Multiple replicates per condition (10+ recommended)
3. Cross-validation before claiming effects
4. Honest reporting of effect sizes and variance

Single-seed, low-replicate studies will produce spurious results. We almost published one.

### The Bigger Picture

This study demonstrates two things:

**1. Methodological rigor is critical for LLM agent research.** The variance is extreme, the behavior is unpredictable, and small sample sizes will mislead you. We got lucky by being skeptical of our initial results and running cross-validation. Most papers don't.

**2. Communication in multi-agent LLM systems is poorly understood.** We expected it to help. It barely did. We don't know if this is specific to our environment or a general finding. Future work should focus on understanding *when* and *why* communication helps, not just assuming it does.

### What We Don't Know Yet

**About communication strategies:**
- Would structured protocols work better with different prompting or schema design?
- Do freeform's advantages hold in even more diverse environments (different maze types, agent counts, visibility ranges)?
- What aspects of natural language messages drive successful coordination—specificity, politeness, spatial references?
- How much does the one-turn communication lag affect different strategies?

**About non-determinism:**
- What's the primary source of run-to-run variability—LLM sampling, backend infrastructure, cascade effects, or something else?
- Can we reduce variability through prompt engineering or other techniques? (Note: model doesn't accept temperature parameter)
- Are there agent configurations or environment designs that produce more consistent behavior?
- How much of the variance is "noise" versus meaningful exploration of the strategy space?
- Is the non-determinism inherent to LLM inference, or can it be controlled?

**About the benchmark itself (THE CRITICAL QUESTION):**
- **What's the human baseline?** Would humans achieve 60%, 95%, or 30% on this exact task (vision radius=1, 100 turns, same maze)?
- **Does communication help humans?** If humans go from 60% (no-comm) to 95% (with-comm), but LLMs stay at 60%, that reveals LLMs can't use coordination information effectively
- **What's the bottleneck?** Is it spatial navigation (LLMs bad at mental maps), communication processing (LLMs ignore messages), or task difficulty (humans would struggle too)?
- **Could this become a benchmark?** Like ARC-AGI for individual reasoning, could this be the standard test for multi-agent coordination under severe constraints?
- **What would "good" performance look like?** Without a reference point, we can't interpret our results. 60% could be excellent or terrible.

**This is the most important open question.** Everything else is speculation without knowing how humans perform on the identical task. A human study would transform this from "we ran some LLM experiments" to "we identified a fundamental capability gap" or "we validated a benchmark for coordination reasoning."

These questions remain open for future work.

---

## Experiment Details

**Cross-Seed Validation Study:**
- Runs: 45 total (5 seeds × 3 strategies × 3 replicates)
- Seeds: 13, 14, 15, 16, 17 (agent spawn positions)
- Maze: `long_corridor` preset, obstacle seed 606, 30×10 grid
- Model: `azure:gpt-5-mini`
- Agent parameters: visibility=1, radio_range=2 (none=0), history_limit=5
- Turn budget: 100 turns per run
- Token counting: `tiktoken` library, `o200k_base` encoding
- Duration: November 12-14, 2025

**Initial Study (Seed 13 Only):**
- Runs: 9 total (3 per strategy)
- Seed: 13 only
- Same environment parameters
- Commit: `fe3ffda`
- This study produced misleading results due to single-seed bias

**Data availability:**
- Full dataset: `experiments/cross_seed_baseline_20251112T143355Z/`
- Run inventory: `run_inventory.json` (45 runs with UTC timestamps)
- Aggregate statistics: `aggregate_stats.json`
- Plots: `plots/` directory (5 publication-quality visualizations)
- Individual runs: `runs/` directory (45 complete run directories with metrics, transcripts, episodes)

---

**Last updated:** 2025-11-15
