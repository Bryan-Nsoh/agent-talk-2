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
        oracle_enabled: bool,
    ) -> None:
        self.model_id = model_id
        self.strategy = strategy
        self.loop_guidance = loop_guidance
        self.history_limit = max(1, history_limit)
        self.radio_range = max(0, radio_range)
        self.capabilities = resolve_strategy_capabilities(strategy, oracle_enabled)
        self.oracle_enabled = self.capabilities.allow_oracle
        self.unified = UnifiedLLM()
        self._wire_decision_model = build_decision_model(strategy, oracle_enabled)

    def _strategy_block(self) -> str:
        strategy = self.strategy.lower()

        general_rules = [
            "Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.",
            "If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.",
            "Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.",
        ]

        if strategy == "none":
            strategy_rules = ["Communication disabled; do not choose COMMUNICATE."]
        elif strategy == "structured":
            strategy_rules = [
                "Allowed: INTENT, REQUEST(YIELD|GUIDE target=(x,y)), HERE. One message max per turn.",
                "Merge trigger: contended_neighbors in intended dir, or prior BLOCK_AGENT/SWAP on same target, or a visible peer within 2 steps would enter your target or swap with you.",
                "Priority: closer to target wins; if equal pick the one whose target reduces Manhattan distance to goal most; if still equal lowest agent_id wins.",
                "On trigger: if you have priority send REQUEST(YIELD,target=T); else send INTENT(MOVE_* or STAY).",
                "Receiver: if REQUEST(YIELD@T) matches your target or swap target, yield exactly 1 turn; if conflicting INTENT and you lack priority, yield 1 turn.",
                "Exit: when you first see G send REQUEST(GUIDE,target=(gx,gy)) once; optionally HERE next turn if on or adjacent to G; do not repeat GUIDE within last 5 turns.",
            ]
        elif strategy == "freeform":
            strategy_rules = [
                "Allowed: one CHAT (<=96 chars) per turn using one of: 'CLAIM@(x,y) D=N|E|S|W T=1', 'YIELD@(x,y) T=1', 'INTENT:N|E|S|W|STAY', 'EXIT@(gx,gy)', 'HERE@(x,y)'.",
                "Use the same trigger, priority, and receiver policy as structured. Parse leniently by substring; ignore malformed lines.",
                "Exit handling identical to structured.",
            ]
        else:
            strategy_rules = ["Communication rules unspecified; default to MOVE and avoid COMMUNICATE."]

        lines = general_rules + strategy_rules
        rules = "\n".join(f"- {line}" for line in lines)
        return f"COMMUNICATION_RULES:\n{rules}\n\n"

    def _loop_block(self) -> str:
        lines = [
            f"Loop monitor: observation history only includes the last {self.history_limit} turns.",
        ]
        if self.loop_guidance.lower() == "active":
            lines.append(
                "If history.loop >= 3 or you have toggled between the same cells repeatedly, change axis or choose a different safe action (STAY, mark, or explore a new direction) before repeating the same move."
            )
            lines.append(
                "Optionally communicate your intent when breaking a loop so nearby agents can coordinate."
            )
        elif self.loop_guidance.lower() == "explore":
            lines.extend(
                [
                    "If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.",
                    "Going away from the goal is acceptable when escaping traps—prioritise clearing the congestion first, then re-approach.",
                    "Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.",
                    "Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.",
                ]
            )
        return "LOOP_RULES:\n" + "\n".join(f"- {line}" for line in lines) + "\n\n"

    def _prompt_for(self, observation: Observation) -> str:
        payload = observation.model_dump(mode="json")
        header = build_prompt_header(
            radio_range=self.radio_range,
            oracle_enabled=self.oracle_enabled,
            action_kinds=self.capabilities.action_kinds,
        ).replace(
            "<OBSERVATION_JSON>\n",
            f"{self._strategy_block()}{self._loop_block()}<OBSERVATION_JSON>\n",
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
