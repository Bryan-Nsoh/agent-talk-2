"""High-level helper for legacy call sites that prefer a functional API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from bioqueryous.llm_clients.unified_llm import UnifiedLLM


_client: Optional[UnifiedLLM] = None


def _get_client(config_path: Optional[str | Path] = None) -> UnifiedLLM:
    global _client
    if config_path is not None:
        return UnifiedLLM(config_path=config_path)
    if _client is None:
        _client = UnifiedLLM()
    return _client


async def make_llm_call(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float | None = None,
    timeout_s: int = 300,
    provider_preference: Optional[str] = None,
    response_format: Optional[Type[BaseModel]] = None,
    config_path: Optional[str | Path] = None,
) -> Any:
    client = _get_client(config_path)
    result, _, _ = await client.run(
        messages,
        model=model,
        provider_preference=provider_preference,
        temperature=temperature,
        output_schema=response_format,
        timeout_s=timeout_s,
    )
    return result


__all__ = ["make_llm_call", "_get_client"]
