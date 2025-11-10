# Direct Grid Live — Tier‑2 Comms Validation

**Last updated:** 2025-11-10T00:16:10Z
**Status:** ?running
**Outcome:** -
**Started:** 2025-11-10

## Question

Do Tier‑2 communication improvements (radio awareness + message memory) reduce out‑of‑range spam and produce actually-heard, useful messages on the 15×5 direct_grid map? How do structured vs freeform compare on quick micro runs?

## Why This Matters

Our assignment is about communication. Prior runs showed agents often “talked into the void.” We added:
- any_peer_in_range and radio_peers_count to Observation (radio awareness)
- recent_messages (last 10 message briefs with ages)
- messages_sent counted only when a recipient actually received the message
- delivered per turn in episode_stream

This experiment validates those changes on a tiny maze where progress should be visible in minutes, not hours.

## Setup

- Model: azure:gpt-5-mini
- Agents: 2
- Map: direct_grid (15×5, seed 13)
- Visibility: R = 1; Radio: r = 2 (local arm)
- Turns: 60
- Strategies:
  - structured (local): INTENT, REQUEST(YIELD|GUIDE), HERE — proactive but in‑range; de‑dup ~5 turns
  - freeform (local): 1 CHAT/turn, exploration‑oriented, in‑range; de‑dup ~5 turns
- Observation extras: any_peer_in_range, radio_peers_count, recent_messages (10 briefs)
- Delivery semantics: messages_sent increments only if recipients>0; episode_stream includes "delivered": N

## Runs

| Run | Started | Status | Notes |
|-----|---------|--------|-------|
| [structured_20251110T001610Z](./runs/structured_20251110T001610Z/) | 2025‑11‑10 00:16 | ?running | Tier‑2, in‑range, exploration prompts minimal |
| [freeform_20251110T001610Z](./runs/freeform_20251110T001610Z/) | 2025‑11‑10 00:16 | ?running | Tier‑2, in‑range, exploration‑oriented freeform |

Planned (next): freeform‑global‑explore arm on the same map.

## What to Inspect (fast)

- episode_stream.jsonl → per frame: "delivered": N (messages heard that turn)
- results/episode.json → render GIFs (title: strategy + “delivered/turn”)
- results/metrics.json → messages_sent, collisions, turns (once complete)

## Results (placeholder)

Add coverage (unique cells visited), time‑to‑first GOAL share, delivered/turn curve, and LAAS once runs finish.

## Interpretation (placeholder)

Assess whether Tier‑2 gating yields heard, informative messages (vs none). Compare structured vs freeform for exploration utility on a small grid.

## Next Steps

- [ ] Add freeform global‑explore arm on direct_grid
- [ ] Render and embed GIFs (structured vs freeform vs freeform‑global)
- [ ] Summarize coverage, time‑to‑GOAL share, LAAS, delivered/turn

