"""LLM-backed policy that produces structured actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from bioqueryous.llm_clients.unified_llm import UnifiedLLM
from llmgrid.prompts import STATIC_HEADER
from llmgrid.schema import Decision, Observation


@dataclass
class DecisionTrace:
    """Structured decision bundle that includes the raw prompt (trace messages optional)."""

    decision: Decision
    prompt: str
    trace_messages: List[dict]


class LlmPolicy:
    """Async wrapper that turns observations into structured decisions via UnifiedLLM."""

    def __init__(self, model_id: str, *, strategy: str, loop_guidance: str, history_limit: int) -> None:
        self.model_id = model_id
        self.strategy = strategy
        self.loop_guidance = loop_guidance
        self.history_limit = max(1, history_limit)
        self.unified = UnifiedLLM()

    def _strategy_block(self) -> str:
        strategy = self.strategy.lower()

        general_rules = [
            "Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.",
            "If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.",
            "Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.",
        ]

        if strategy == "none":
            strategy_rules = ["Communication disabled; do not choose COMMUNICATE."]
        elif strategy == "intent":
            strategy_rules = [
                "If a neighbor within 2 cells could collide with your intended move (same target cell or swap) next turn, COMMUNICATE exactly one INTENT now.",
                "If you receive an INTENT for the same target or swap and your agent_id is lexicographically larger, yield on the next turn (STAY or a safe alternate); otherwise MOVE.",
                "When no conflict is likely, MOVE and skip communication.",
            ]
        elif strategy == "negotiation":
            strategy_rules = [
                "Allowed messages: HERE, INTENT, SENSE, REQUEST(YIELD|GUIDE|MEET); send at most one message when you COMMUNICATE.",
                "Use the same conflict trigger as INTENT; prefer INTENT unless another message resolves the risk more clearly.",
                "When no conflict is likely, MOVE and skip communication.",
            ]
        elif strategy == "freeform":
            strategy_rules = [
                "If a neighbor within 2 cells could collide with your intended move next turn, COMMUNICATE one <=96-char sentence with your plan and a simple request.",
                "When no conflict is likely, MOVE and do not communicate.",
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
        header = STATIC_HEADER.replace(
            "<OBSERVATION_JSON>\n",
            f"{self._strategy_block()}{self._loop_block()}<OBSERVATION_JSON>\n",
            1,
        )
        return f"{header}{json.dumps(payload, separators=(',', ':'))}\n</OBSERVATION_JSON>"

    async def decide_async(self, observation: Observation) -> Decision:
        prompt = self._prompt_for(observation)
        decision, _, _ = await self.unified.run(
            [{"role": "user", "content": prompt}],
            model=self.model_id,
            output_schema=Decision,
        )
        return decision

    async def decide_with_trace_async(self, observation: Observation) -> DecisionTrace:
        prompt = self._prompt_for(observation)
        decision, _, _ = await self.unified.run(
            [{"role": "user", "content": prompt}],
            model=self.model_id,
            output_schema=Decision,
        )
        return DecisionTrace(decision=decision, prompt=prompt, trace_messages=[])

    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - guard rail
        raise RuntimeError(
            "LlmPolicy.decide() is disabled; use decide_async() within the episode event loop."
        )

    def decide_with_trace(self, observation: Observation) -> DecisionTrace:  # pragma: no cover - guard rail
        raise RuntimeError(
            "LlmPolicy.decide_with_trace() is disabled; use decide_with_trace_async()."
        )
