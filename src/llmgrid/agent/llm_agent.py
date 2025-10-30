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

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.unified = UnifiedLLM()

    def _prompt_for(self, observation: Observation) -> str:
        payload = observation.model_dump(mode="json")
        return f"{STATIC_HEADER}{json.dumps(payload, separators=(',', ':'))}\n</OBSERVATION_JSON>"

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
