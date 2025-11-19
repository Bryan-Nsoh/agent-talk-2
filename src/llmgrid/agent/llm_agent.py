"""LLM-backed policy that produces structured actions for the map-sharing environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from llmgrid.llm_clients.unified_llm import UnifiedLLM
from llmgrid.prompts import build_prompt_header
from llmgrid.schema import Decision, Observation, build_decision_model, coerce_decision, resolve_strategy_capabilities


@dataclass
class DecisionTrace:
    decision: Decision
    prompt: str
    trace_messages: List[dict]


class LlmPolicy:
    def __init__(
        self,
        model_id: str,
        *,
        strategy: str,
        loop_guidance: str,
        history_limit: int,
        radio_range: int,
    ) -> None:
        self.model_id = model_id
        self.strategy = strategy
        self.loop_guidance = loop_guidance
        self.history_limit = history_limit
        self.radio_range = radio_range
        self.capabilities = resolve_strategy_capabilities(strategy)
        self.unified = UnifiedLLM()
        self._wire_decision_model = build_decision_model(strategy)

    def _prompt_for(self, observation: Observation) -> str:
        payload = observation.model_dump(mode="json")
        header = build_prompt_header()
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
        raise RuntimeError("Use decide_async() within the async loop.")

    def decide_with_trace(self, observation: Observation) -> DecisionTrace:  # pragma: no cover - guard rail
        raise RuntimeError("Use decide_with_trace_async().")
