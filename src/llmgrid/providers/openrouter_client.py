"""Utilities for constructing Pydantic-AI agents backed by OpenRouter."""

from __future__ import annotations

import os
from typing import Type

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider


def build_agent(model_id: str, system_prompt: str, output_type: Type[BaseModel]) -> Agent:
    """Create a structured-output agent bound to OpenRouter.

    The function fails fast when credentials are missing or the caller does not
    request an OpenRouter-qualified model, matching the AGENTS.md directives.
    """

    if not model_id:
        raise ValueError("Model id must be provided.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.env, reload your shell, and retry."
        )

    # OpenRouterProvider configures the base URL and headers for OpenRouter.
    provider = OpenRouterProvider(api_key=api_key)
    model = OpenAIChatModel(model_id, provider=provider)

    # Parameter minimalism: rely on provider defaults, no temperature or token caps.
    return Agent(model=model, system_prompt=system_prompt, output_type=output_type)
