"""LLM-backed policy that produces structured actions."""

from __future__ import annotations

import json
from typing import Optional

import json
from dataclasses import dataclass
from typing import List

from llmgrid.prompts import STATIC_HEADER
from llmgrid.providers.openrouter_client import build_agent
from llmgrid.schema import Decision, Observation


@dataclass
class DecisionTrace:
    """Structured decision bundle that includes the raw prompt and model trace."""

    decision: Decision
    prompt: str
    trace_messages: List[dict]


class LlmPolicy:
    """Small wrapper that turns observations into structured decisions."""

    def __init__(self, model_id: str) -> None:
        if model_id.startswith("openrouter:"):
            # Allow callers to pass the fully qualified identifier.
            model_name = model_id.split(":", 1)[1]
        else:
            model_name = model_id
        self.agent = build_agent(
            model_id=model_name,
            system_prompt=STATIC_HEADER,
            output_type=Decision,
        )

    def decide(self, observation: Observation) -> Decision:
        payload = observation.model_dump(mode="json")
        prompt = f"{STATIC_HEADER}{json.dumps(payload, separators=(',', ':'))}\n</OBSERVATION_JSON>"
        result = self.agent.run_sync(prompt)
        return result.output

    def decide_with_trace(self, observation: Observation) -> DecisionTrace:
        """Return the decision together with the exact prompt and model trace."""
        payload = observation.model_dump(mode="json")
        prompt = f"{STATIC_HEADER}{json.dumps(payload, separators=(',', ':'))}\n</OBSERVATION_JSON>"
        result = self.agent.run_sync(prompt)
        raw_messages = result.all_messages_json()
        if isinstance(raw_messages, bytes):
            trace_messages = json.loads(raw_messages.decode("utf-8"))
        else:
            trace_messages = raw_messages  # already JSON-compatible
        return DecisionTrace(decision=result.output, prompt=prompt, trace_messages=trace_messages)
