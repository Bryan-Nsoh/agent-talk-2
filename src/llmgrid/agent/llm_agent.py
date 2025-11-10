"""LLM-backed policy that produces structured actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from llmgrid.llm_clients.unified_llm import UnifiedLLM
from llmgrid.prompts import build_prompt_header
from llmgrid.schema import (
    Decision,
    Observation,
    build_decision_model,
    coerce_decision,
    resolve_strategy_capabilities,
)


@dataclass
class DecisionTrace:
    """Structured decision bundle that includes the raw prompt (trace messages optional)."""

    decision: Decision
    prompt: str
    trace_messages: List[dict]


class LlmPolicy:
    """Async wrapper that turns observations into structured decisions via UnifiedLLM."""

    def __init__(
        self,
        model_id: str,
        *,
        strategy: str,
        loop_guidance: str,
        history_limit: int,
        radio_range: int,
        freeform_global: bool = False,
    ) -> None:
        self.model_id = model_id
        self.strategy = strategy
        self.loop_guidance = loop_guidance
        self.history_limit = max(1, history_limit)
        self.radio_range = max(0, radio_range)
        self.capabilities = resolve_strategy_capabilities(strategy, False)
        self.oracle_enabled = False
        self.freeform_global = freeform_global
        self.unified = UnifiedLLM()
        self._wire_decision_model = build_decision_model(strategy, False)

    def _strategy_block(self) -> str:
        strategy = self.strategy.lower()

        general_rules = [
            "Keep comments ≤25 words and use them to explain your reasoning (mention coordinates or map features when possible).",
            "If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.",
            "Treat CONTENDED neighbors as high risk: only enter if no safer option, and communicate or yield when you do.",
        ]

        if strategy == "none":
            strategy_rules = ["Communication disabled; do not choose COMMUNICATE."]
        elif strategy == "structured":
            strategy_rules = [
                "DEFAULT TO MOVE. Only COMMUNICATE when the message prevents imminent collision or shares critical info (goal location, essential map gap, stuck peer).",
                "Allowed: INTENT, REQUEST(YIELD|GUIDE target=(x,y)), HERE, MAP_REQUEST(origin=(x,y),radius=2). One message max per turn.",
                "When to communicate: only if any_peer_in_range is true and you have useful info (collision risk, new corridor, map gap) that a nearby peer benefits from.",
                "Good reasons: approaching a shared cell, you see G, you discovered a useful corridor or dead end, your buddy might be stuck, or you need a map snippet to progress.",
                "Priority: when 2+ agents want the same cell, LOWEST agent_id MOVES immediately (no announcement needed). Higher IDs MUST yield (stay/reroute). No mutual yielding, no wasted turns announcing priority.",
                "MAP_REQUEST returns either MAP_PATCH (5×5 snippet) or MAP_NO_PATCH. Use MAP_REQUEST when `nearest_frontier` stays unchanged for several turns—include that coordinate in the origin field.",
                "Message choice: INTENT to share your next move; REQUEST(YIELD,target=T) if you need priority; REQUEST(GUIDE,target=(gx,gy)) to share G; HERE to confirm your position; MAP_REQUEST to fetch a snippet when stuck.",
                "MAP_PATCH arrives automatically—treat it as authoritative and cite the new coordinates in your comment. Avoid repeats: do not send the same content within 5 turns unless new information appeared.",
            ]
        elif strategy == "freeform":
            strategy_rules = [
                "DEFAULT TO MOVE. Only CHAT when the message prevents imminent collision or shares critical info (goal location, dead end, you're rerouting around a peer).",
                "Allowed: one CHAT (<=96 chars) per turn. Write naturally to help your teammate.",
                ("When to communicate: share new useful info each turn; radio has unlimited range."
                 if self.freeform_global else
                 "When to communicate: only if any_peer_in_range is true. Share something useful (new route, goal location, dead end you verified, you are rerouting, or you are stuck)."),
                "Use coordinates so teammates can mark their maps: e.g., 'heading east toward (5,2)', 'found goal at (14,4)', 'dead end north; trying south', 'sharing loop at (3,1)-(3,2)'.",
                "Priority: when 2+ agents want the same cell, LOWEST agent_id goes first. Higher IDs yield. Example: 'I'm a5, yielding (5,5) to you, going west' or just move without announcing if you're yielding.",
                "Be cooperative and concise; avoid repeating unchanged info within ~5 turns.",
            ]
        else:
            strategy_rules = ["Communication rules unspecified; default to MOVE and avoid COMMUNICATE."]

        lines = general_rules + strategy_rules
        rules = "\n".join(f"- {line}" for line in lines)
        return f"COMMUNICATION_RULES:\n{rules}\n\n"

    def _prompt_for(self, observation: Observation) -> str:
        payload = observation.model_dump(mode="json")
        header = build_prompt_header(
            radio_range=self.radio_range,
            action_kinds=self.capabilities.action_kinds,
        ).replace(
            "<OBSERVATION_JSON>\n",
            f"{self._strategy_block()}<OBSERVATION_JSON>\n",
            1,
        )
        return f"{header}{json.dumps(payload, separators=(',', ':'))}\n</OBSERVATION_JSON>"

    async def decide_async(self, observation: Observation) -> Decision:
        prompt = self._prompt_for(observation)
        wire_decision, _, _ = await self.unified.run(
            [{"role": "user", "content": prompt}],
            model=self.model_id,
            output_schema=self._wire_decision_model,
            max_spatial_retries=3,
        )
        return coerce_decision(wire_decision)

    async def decide_with_trace_async(self, observation: Observation) -> DecisionTrace:
        prompt = self._prompt_for(observation)
        wire_decision, _, _ = await self.unified.run(
            [{"role": "user", "content": prompt}],
            model=self.model_id,
            output_schema=self._wire_decision_model,
            max_spatial_retries=3,
        )
        decision = coerce_decision(wire_decision)
        return DecisionTrace(decision=decision, prompt=prompt, trace_messages=[])

    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - guard rail
        raise RuntimeError(
            "LlmPolicy.decide() is disabled; use decide_async() within the episode event loop."
        )

    def decide_with_trace(self, observation: Observation) -> DecisionTrace:  # pragma: no cover - guard rail
        raise RuntimeError(
            "LlmPolicy.decide_with_trace() is disabled; use decide_with_trace_async()."
        )
