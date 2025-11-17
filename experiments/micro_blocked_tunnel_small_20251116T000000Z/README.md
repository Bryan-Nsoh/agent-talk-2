# Micro Blocked Tunnel (10x7)

**Last updated:** 2025-11-16T14:45:00Z  
**Status:** blocked (no LLM runs completed; network to Azure appears restricted)  
**Outcome:** -  
**Started:** 2025-11-16

## Question

Which lightweight communication style (none, structured, freeform, hybrid-tag, hybrid-key=value) best handles a tiny blocked-tunnel maze with 2 agents and 50 turns?

## Setup

- Maze: `micro_blocked_tunnel_small` (10x7), goal at (9,6), starts a1=(0,0), a2=(0,6)  
- Agents: 2  
- Turns: 50  
- Model: openrouter:openai/gpt-5-mini; same for all baselines  
- Visibility, radio_range: use CLI defaults unless baseline specifies otherwise  
- Seeds: 5 (0–4)  
- Logging: `--log-prompts --log-movements --emit-config`

## Baselines

See `baselines.json` for the 12 planned variants covering:
- comm_strategy: none / structured / freeform / hybrid-tag / hybrid-key=value
- safety bias: default vs cautious
- priority rules: lowest-id goes vs yields
- message discipline: send-on-block vs choke-only

## Runs

Planned: 12 baselines × 5 seeds = 60 episodes. Not yet executed.

## Results

Blocked: No runs completed yet. Azure gpt-5-mini calls hang (network restricted). Runs directory cleaned; ready to rerun once outbound access is available. See `docs/lag_deep_dive.md` for current state and rerun instructions.

## Next Steps

- Run the full sweep via `scripts/run_micro_blocked_tunnel_small.sh`.
- Aggregate metrics (success, collisions, messages).
- Deep-dive top 2–3 with message helpful/harmful analysis.
