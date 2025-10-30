"""Unified LLM client with provider-aware rate limiting and pooling.

This module is a trimmed-down adaptation of the production BioQueryous orchestration
stack.  It keeps the core scheduling behaviour—Azure/OpenAI aware pooling,
token-per-minute budgeting, and adaptive backoff—while depending only on modules
that live inside this repository.  The aim is to provide a ready-to-use client
for experimentation without dragging in the full co-scientist codebase.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import httpx
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

try:  # pragma: no cover - optional dependency
    from pydantic_ai.providers.azure import AzureProvider
except Exception:  # pragma: no cover
    AzureProvider = None  # type: ignore[assignment]

from bioqueryous.utils.errors import APIError, CircuitBreakerOpenError
from bioqueryous.utils.real_time_logger import get_logger

LOGGER = get_logger()

T = TypeVar("T", bound=BaseModel)


@contextmanager
def _temp_env(**kwargs: Optional[str]):
    """Temporarily set environment variables for downstream SDKs."""

    old_values: Dict[str, Optional[str]] = {}
    for key, value in kwargs.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


DEFAULT_CONFIG_LOCATIONS: List[Path] = [
    Path(os.environ.get("BIOQUERYOUS_MODELS_CONFIG", "")),
    Path.cwd() / "models.yaml",
    Path.home() / ".bioqueryous" / "models.yaml",
]
DEFAULT_CONFIG_LOCATIONS = [p for p in DEFAULT_CONFIG_LOCATIONS if str(p).strip()]


@dataclass
class ProvidersConfig:
    raw: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PoolsConfig:
    raw: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class PricingConfig:
    raw: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class FullConfig:
    providers: ProvidersConfig
    pools: PoolsConfig
    pricing: PricingConfig


def load_config(config_path: Optional[Union[str, Path]] = None) -> Tuple[FullConfig, Dict[str, Any]]:
    """Load the models.yaml configuration."""

    search_locations = DEFAULT_CONFIG_LOCATIONS
    if config_path is not None:
        search_locations = [Path(config_path)] + search_locations

    config_file = _first_existing(search_locations)
    if config_file is None:
        raise FileNotFoundError(
            "models.yaml not found. Set BIOQUERYOUS_MODELS_CONFIG or place the file in the repo root or ~/.bioqueryous/."
        )

    with config_file.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    providers = ProvidersConfig(raw=config_data.get("providers", {}))
    pools = PoolsConfig(raw=config_data.get("model_pools", {}))
    pricing = PricingConfig(raw=config_data.get("model_pricing", {}))
    limits = config_data.get("limits", {})

    LOGGER.info("[ModelConfig] Loaded configuration from %s", config_file)
    return FullConfig(providers=providers, pools=pools, pricing=pricing), limits


@dataclass
class CallRecord:
    timestamp: float
    provider: str
    model: str
    in_tokens: int
    out_tokens: int


@dataclass
class TokenStats:
    calls: List[CallRecord] = field(default_factory=list)
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)


class TokenTracker:
    """Simple in-memory accounting for per-model token usage."""

    def __init__(self, pricing: PricingConfig) -> None:
        self._stats = TokenStats()
        self._pricing = pricing.raw

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        call = CallRecord(time.time(), provider, model, input_tokens, output_tokens)
        self._stats.calls.append(call)
        model_stats = self._stats.by_model.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        model_stats["input"] += input_tokens
        model_stats["output"] += output_tokens
        model_stats["calls"] += 1
        LOGGER.debug(
            "[tokens] %s:%s | in=%s out=%s totals in=%s out=%s",
            provider,
            model,
            input_tokens,
            output_tokens,
            model_stats["input"],
            model_stats["output"],
        )

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {model: stats.copy() for model, stats in self._stats.by_model.items()}


class RateLimiter:
    """Provider/model limiter with per-minute token budgets and adaptive utilisation."""

    def __init__(self, limits_cfg: Dict[str, Any]) -> None:
        self._cfg = limits_cfg or {}
        self._defaults = self._cfg.get("defaults", {})
        self._provider_overrides = self._cfg.get("providers", {})
        self._model_overrides = self._cfg.get("models", {})

        adaptive_cfg = self._cfg.get("adaptive", {})
        self._adaptive_enabled = adaptive_cfg.get("enabled", True)
        self._initial_util = adaptive_cfg.get("initial_utilization", 0.85)
        self._reduction_factor = adaptive_cfg.get("reduction_factor", 0.8)
        self._recovery_increment = adaptive_cfg.get("recovery_increment_pct", 0.05)
        self._recovery_interval = adaptive_cfg.get("recovery_interval_s", 60.0)
        self._cooldown = adaptive_cfg.get("cooldown_s", 30.0)
        self._min_util = adaptive_cfg.get("min_utilization", 0.5)
        self._max_util = adaptive_cfg.get("max_utilization", 0.95)

        self._current_util: Dict[str, float] = {}
        self._last_reduction: Dict[str, float] = {}
        self._last_recovery: Dict[str, float] = {}

        self._locks: Dict[tuple[str, int], asyncio.Semaphore] = {}
        self._lock_targets: Dict[tuple[str, int], int] = {}
        self._rpm_window: Dict[str, deque] = {}
        self._tpm_window: Dict[str, deque] = {}

    # --- configuration helpers -------------------------------------------------
    def _provider_cfg(self, provider: str) -> Dict[str, Any]:
        return {**self._defaults, **self._provider_overrides.get(provider, {})}

    def _model_cfg(self, provider: str, model: str) -> Dict[str, Any]:
        key = f"{provider}:{model}"
        return {**self._provider_cfg(provider), **self._model_overrides.get(key, {})}

    def _effective_tpm(self, provider: str, configured_tpm: int) -> int:
        if not self._adaptive_enabled:
            return configured_tpm
        util = self._current_util.get(provider, self._initial_util)
        return int(configured_tpm * util)

    # --- adaptive controls -----------------------------------------------------
    def on_rate_limit(self, provider: str) -> None:
        if not self._adaptive_enabled:
            return
        now = time.time()
        if now - self._last_reduction.get(provider, 0.0) < self._cooldown:
            return
        current = self._current_util.get(provider, self._initial_util)
        new_util = max(current * self._reduction_factor, self._min_util)
        self._current_util[provider] = new_util
        self._last_reduction[provider] = now
        LOGGER.warning(
            "[limiter] reduced utilisation for %s: %.0f%% -> %.0f%%",
            provider,
            current * 100,
            new_util * 100,
        )

    def on_success(self, provider: str) -> None:
        if not self._adaptive_enabled:
            return
        now = time.time()
        if now - self._last_recovery.get(provider, 0.0) < self._recovery_interval:
            return
        current = self._current_util.get(provider, self._initial_util)
        if current >= self._max_util:
            return
        new_util = min(current + self._recovery_increment, self._max_util)
        self._current_util[provider] = new_util
        self._last_recovery[provider] = now
        LOGGER.info(
            "[limiter] increased utilisation for %s: %.0f%% -> %.0f%%",
            provider,
            current * 100,
            new_util * 100,
        )

    # --- resource acquisition --------------------------------------------------
    def _lock_for(self, provider: str) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = (provider, id(loop))
        cfg = self._provider_cfg(provider)
        target = int(cfg.get("concurrency", 4))
        sem = self._locks.get(key)
        if sem is None or self._lock_targets.get(key) != target:
            sem = asyncio.Semaphore(target)
            self._locks[key] = sem
            self._lock_targets[key] = target
        return sem

    def estimate_tokens(self, provider: str, model: str) -> int:
        cfg = self._model_cfg(provider, model)
        return int(cfg.get("estimate_tokens_per_call", 0))

    async def acquire(self, provider: str, model: str, estimated_tokens: int) -> asyncio.Callable[[Optional[int]], None]:
        sem = self._lock_for(provider)
        await sem.acquire()
        try:
            now = time.time()
            # RPM window
            rpm_cfg = self._model_cfg(provider, model).get("rpm")
            window = self._rpm_window.setdefault(provider, deque())
            if rpm_cfg:
                while window and now - window[0] > 60.0:
                    window.popleft()
                while len(window) >= int(rpm_cfg):
                    sleep_for = 60.0 - (now - window[0])
                    await asyncio.sleep(max(0.0, sleep_for))
                    now = time.time()
                    while window and now - window[0] > 60.0:
                        window.popleft()
                window.append(now)

            # TPM window
            tpm_cfg = self._model_cfg(provider, model).get("tpm")
            token_window = self._tpm_window.setdefault(provider, deque())
            if tpm_cfg and estimated_tokens:
                while token_window and now - token_window[0][0] > 60.0:
                    token_window.popleft()
                effective_tpm = self._effective_tpm(provider, int(tpm_cfg))
                used = sum(tokens for _, tokens in token_window)
                while used + estimated_tokens > effective_tpm:
                    sleep_for = 60.0 - (now - token_window[0][0]) if token_window else 0.25
                    await asyncio.sleep(max(0.0, sleep_for))
                    now = time.time()
                    while token_window and now - token_window[0][0] > 60.0:
                        token_window.popleft()
                    used = sum(tokens for _, tokens in token_window)
                token_window.append((now, estimated_tokens))
        except Exception:
            sem.release()
            raise

        async def release(actual_tokens: Optional[int]) -> None:
            if actual_tokens is not None and actual_tokens > estimated_tokens:
                token_window = self._tpm_window.setdefault(provider, deque())
                token_window.append((time.time(), actual_tokens - estimated_tokens))
            sem.release()

        return release


class ModelPool:
    """Round-robin pool with failure tracking for model rotations."""

    def __init__(self, pools_cfg: PoolsConfig, *, fail_ttl: float = 60.0, provider_threshold: int = 3, provider_cooldown: float = 90.0) -> None:
        self._pools = pools_cfg.raw
        self._lock = asyncio.Lock()
        self._rr: Dict[str, int] = {}
        self._fail_ttl = fail_ttl
        self._provider_threshold = provider_threshold
        self._provider_cooldown = provider_cooldown
        self._model_fail_until: Dict[str, float] = {}
        self._provider_fail_count: Dict[str, int] = {}
        self._provider_open_until: Dict[str, float] = {}

    def _fail_key(self, provider: str, target: str) -> str:
        return f"{provider}::{target}"

    def _is_model_failed(self, provider: str, target: str) -> bool:
        key = self._fail_key(provider, target)
        until = self._model_fail_until.get(key)
        if not until:
            return False
        if time.time() >= until:
            self._model_fail_until.pop(key, None)
            return False
        return True

    def _is_provider_open(self, provider: str) -> bool:
        until = self._provider_open_until.get(provider)
        if not until:
            return False
        if time.time() >= until:
            self._provider_open_until.pop(provider, None)
            self._provider_fail_count.pop(provider, None)
            return False
        return True

    async def mark_success(self, provider: str) -> None:
        async with self._lock:
            self._provider_fail_count[provider] = 0

    async def mark_failure(self, provider: str, target: str) -> None:
        async with self._lock:
            self._model_fail_until[self._fail_key(provider, target)] = time.time() + self._fail_ttl
            count = self._provider_fail_count.get(provider, 0) + 1
            self._provider_fail_count[provider] = count
            if count >= self._provider_threshold:
                self._provider_open_until[provider] = time.time() + self._provider_cooldown
                LOGGER.warning("[pool] circuit opened for provider %s", provider)

    async def select(self, model_key: str, provider_preference: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        pool_entries = self._pools.get(model_key)
        if not pool_entries:
            raise ValueError(f"No pool configured for '{model_key}'")

        candidates = list(pool_entries)
        if provider_preference:
            candidates = [entry for entry in candidates if entry.get("provider") == provider_preference]
            if not candidates:
                raise ValueError(f"No entries for '{model_key}' with provider '{provider_preference}'")

        async with self._lock:
            start_idx = self._rr.get(model_key, 0)
            for offset in range(len(candidates)):
                idx = (start_idx + offset) % len(candidates)
                entry = candidates[idx]
                provider = entry["provider"]
                target = entry.get("deployment") or entry.get("model")
                if not target:
                    continue
                if self._is_provider_open(provider):
                    continue
                if self._is_model_failed(provider, target):
                    continue
                self._rr[model_key] = (idx + 1) % len(candidates)
                return provider, dict(entry)
        raise CircuitBreakerOpenError(f"All providers for '{model_key}' are temporarily unavailable")

    @property
    def provider_cooldown(self) -> float:
        return self._provider_cooldown


def _build_azure_provider(cfg: Dict[str, Any]) -> AzureProvider:  # type: ignore[return-value]
    if AzureProvider is None:  # pragma: no cover
        raise RuntimeError("pydantic-ai azure provider not installed")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or cfg.get("base_url") or cfg.get("endpoint")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or cfg.get("api_key")
    api_version = os.getenv("OPENAI_API_VERSION") or cfg.get("api_version") or "2024-10-21"
    return AzureProvider(azure_endpoint=endpoint, api_version=api_version, api_key=api_key)


def _build_openai_provider(cfg: Dict[str, Any]) -> OpenAIProvider:
    base_url = cfg.get("base_url")
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env) or cfg.get("api_key")
    return OpenAIProvider(base_url=base_url, api_key=api_key)


def _build_openrouter_provider(cfg: Dict[str, Any]) -> OpenRouterProvider:
    api_key_env = cfg.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.getenv(api_key_env) or cfg.get("api_key")
    return OpenRouterProvider(api_key=api_key)


class UnifiedLLM:
    """High-level entry point that wraps provider pooling, throttling, and logging."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        cfg, limits = load_config(config_path)
        self._cfg = cfg
        self.pool = ModelPool(cfg.pools)
        self.tokens = TokenTracker(cfg.pricing)
        self.limiter = RateLimiter(limits)

    # ------------------------------------------------------------------ public
    async def run(
        self,
        messages: Union[str, List[Dict[str, str]]],
        *,
        model: str,
        provider_preference: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        output_schema: Optional[Type[T]] = None,
        timeout_s: int = 300,
        max_spatial_retries: int = 3,
    ) -> Tuple[Union[str, T], int, int]:
        attempt = 0
        last_exc: Optional[Exception] = None

        # Resolve pool entry
        provider_name: str
        model_entry: Dict[str, Any]
        if ":" in model:
            provider_name, model_name = model.split(":", 1)
            model_entry = {"provider": provider_name, "model": model_name}
        else:
            provider_name, model_entry = await self.pool.select(model, provider_preference)
            model_name = model_entry.get("deployment") or model_entry.get("model")
            if not model_name:
                raise APIError("Model configuration missing deployment/model name")

        estimated_tokens = self.limiter.estimate_tokens(provider_name, model_name)

        while attempt <= max_spatial_retries:
            attempt += 1
            release = await self.limiter.acquire(provider_name, model_name, estimated_tokens)
            try:
                result, in_tok, out_tok = await self._dispatch(
                    provider=provider_name,
                    model_name=model_name,
                    model_entry=model_entry,
                    messages=messages,
                    temperature=temperature,
                    output_schema=output_schema,
                    timeout_s=timeout_s,
                )
                await release((in_tok or 0) + (out_tok or 0))
                await self.pool.mark_success(provider_name)
                self.limiter.on_success(provider_name)
                return result, in_tok, out_tok
            except Exception as exc:  # pragma: no cover - error path
                await release(None)
                last_exc = exc
                await self.pool.mark_failure(provider_name, model_name)
                error_class = self._classify_error(exc)
                LOGGER.warning(
                    "[llm] %s:%s failed class=%s exception=%s: %s",
                    provider_name,
                    model_name,
                    error_class,
                    type(exc).__name__,
                    str(exc),
                )
                if error_class in {"invalid", "auth"}:
                    break
                if error_class == "rate":
                    self.limiter.on_rate_limit(provider_name)
                if ":" not in model and attempt <= max_spatial_retries:
                    try:
                        provider_name, model_entry = await self.pool.select(model, provider_preference)
                        model_name = model_entry.get("deployment") or model_entry.get("model")
                        estimated_tokens = self.limiter.estimate_tokens(provider_name, model_name)
                        continue
                    except CircuitBreakerOpenError:
                        await asyncio.sleep(self.pool.provider_cooldown)
                        continue
                break

        raise APIError(f"LLM call failed after {attempt} attempts: {last_exc}")

    # ---------------------------------------------------------------- helpers
    async def _dispatch(
        self,
        *,
        provider: str,
        model_name: str,
        model_entry: Dict[str, Any],
        messages: Union[str, List[Dict[str, str]]],
        temperature: Optional[float],
        output_schema: Optional[Type[T]],
        timeout_s: int,
    ) -> Tuple[Union[str, T], int, int]:
        mode = model_entry.get("mode", "agent")
        if mode == "sdk":
            return await self._call_with_sdk(
                provider,
                model_name,
                messages,
                temperature,
                output_schema,
                timeout_s,
                model_entry=model_entry,
            )
        return await self._call_with_agent(
            provider,
            model_name,
            messages,
            temperature,
            output_schema,
            timeout_s,
            model_entry=model_entry,
        )

    # -- agent path -------------------------------------------------------------
    async def _call_with_agent(
        self,
        provider: str,
        model_name: str,
        messages: Union[str, List[Dict[str, str]]],
        temperature: Optional[float],
        output_schema: Optional[Type[T]],
        timeout_s: int,
        *,
        model_entry: Dict[str, Any],
    ) -> Tuple[Union[str, T], int, int]:
        provider_cfg = self._cfg.providers.raw.get(provider, {})
        provider_prefix = {
            "azure": "azure",
            "openrouter": "openrouter",
        }.get(provider.lower(), "openai")
        model_str = f"{provider_prefix}:{model_name}"

        env_overrides: Dict[str, Optional[str]] = {}
        if provider_prefix == "openai":
            base_url = provider_cfg.get("base_url")
            api_key_env = provider_cfg.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env) or provider_cfg.get("api_key")
            env_overrides.update({"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": api_key})
        elif provider_prefix == "azure":
            endpoint = provider_cfg.get("base_url") or provider_cfg.get("endpoint")
            api_key_env = provider_cfg.get("api_key_env", "AZURE_API_KEY")
            api_key = os.getenv(api_key_env) or provider_cfg.get("api_key")
            api_version = provider_cfg.get("api_version", "2024-10-21")
            env_overrides.update(
                {
                    "AZURE_OPENAI_ENDPOINT": endpoint,
                    "AZURE_OPENAI_API_KEY": api_key,
                    "OPENAI_API_VERSION": api_version,
                }
            )

        settings = ModelSettings(temperature=temperature, timeout=timeout_s)
        with _temp_env(**env_overrides):
            prompt = messages if isinstance(messages, str) else _messages_to_prompt(messages)
            agent = Agent(model_str, output_type=output_schema or str, model_settings=settings)
            usage_in = usage_out = 0
            async with agent.iter(prompt) as agent_run:
                async for node in agent_run:
                    response = getattr(node, "model_response", None)
                    if response and response.usage:
                        usage = response.usage
                        usage_in = getattr(usage, "input_tokens", usage_in)
                        usage_out = getattr(usage, "output_tokens", usage_out)
                run_result = agent_run.result
            if run_result is None:
                raise APIError("Agent returned no output")
            output = run_result.output
            if output_schema and not isinstance(output, BaseModel):
                if isinstance(output, str):
                    output = output_schema.model_validate_json(output)
                elif isinstance(output, dict):
                    output = output_schema.model_validate(output)
                else:
                    raise APIError("Unable to coerce agent output to schema")
            self.tokens.record(provider, model_name, usage_in, usage_out)
            return output, usage_in, usage_out

    # -- sdk path ---------------------------------------------------------------
    async def _call_with_sdk(
        self,
        provider: str,
        model_name: str,
        messages: Union[str, List[Dict[str, str]]],
        temperature: Optional[float],
        output_schema: Optional[Type[T]],
        timeout_s: int,
        *,
        model_entry: Dict[str, Any],
    ) -> Tuple[Union[str, T], int, int]:
        message_list = (
            messages
            if isinstance(messages, list)
            else [{"role": "user", "content": str(messages)}]
        )

        if provider.lower() == "azure":
            cfg = self._cfg.providers.raw.get("azure", {})
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or cfg.get("base_url") or cfg.get("endpoint")
            if not endpoint:
                raise APIError("Azure endpoint not configured")
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or cfg.get("api_key")
            if not api_key:
                raise APIError("Azure API key not configured")
            api_version = os.getenv("OPENAI_API_VERSION") or cfg.get("api_version", "2024-10-21")

            if model_entry.get("mode") == "responses":
                return await self._call_azure_responses(
                    endpoint,
                    api_key,
                    model_name,
                    message_list,
                    api_version,
                    timeout_s,
                    model_entry,
                    output_schema,
                )

            base_url = f"{endpoint.rstrip('/')}/openai/deployments/{model_name}"
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=float(timeout_s),
                default_query={"api-version": api_version},
                max_retries=1,
            )
            provider_label = "azure"
        else:
            provider_cfg = self._cfg.providers.raw.get(provider, {})
            base_url = provider_cfg.get("base_url")
            api_key_env = provider_cfg.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env) or provider_cfg.get("api_key")
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=float(timeout_s), max_retries=1)
            provider_label = "openai" if (base_url or "").startswith("https://api.openai.com") else "openai-compatible"

        if output_schema:
            completion = await client.beta.chat.completions.parse(
                model=model_name,
                messages=message_list,
                temperature=temperature,
                response_format=output_schema,
            )
            usage = getattr(completion, "usage", None)
            in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            self.tokens.record(provider_label, model_name, in_tokens, out_tokens)
            return completion.choices[0].message.parsed, in_tokens, out_tokens

        completion = await client.chat.completions.create(
            model=model_name,
            messages=message_list,
            temperature=temperature,
        )
        usage = getattr(completion, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        content = (completion.choices[0].message.content or "").strip()
        self.tokens.record(provider_label, model_name, in_tokens, out_tokens)
        return content, in_tokens, out_tokens

    async def _call_azure_responses(
        self,
        endpoint: str,
        api_key: str,
        model_name: str,
        messages: List[Dict[str, str]],
        api_version: str,
        timeout_s: int,
        model_entry: Dict[str, Any],
        output_schema: Optional[Type[T]],
    ) -> Tuple[Union[str, T], int, int]:
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": _convert_messages(messages),
        }
        if output_schema is not None:
            schema = output_schema.model_json_schema()  # type: ignore[attr-defined]
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": output_schema.__name__, "schema": schema},
            }
        responses_endpoint = model_entry.get("responses_endpoint", "/openai/v1/responses")
        url = f"{endpoint.rstrip('/')}{responses_endpoint}"
        headers = {"api-key": api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=timeout_s, verify=False) as client:
            response = await client.post(url, params={"api-version": api_version}, json=payload)
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)

        outputs = data.get("output", [])
        json_block: Optional[Dict[str, Any]] = None
        text_parts: List[str] = []
        for block in outputs:
            if block.get("type") != "message":
                continue
            for content in block.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text_parts.append(content.get("text", ""))
                if content.get("type") == "json" and isinstance(content.get("json"), dict):
                    json_block = content["json"]

        if output_schema is not None:
            if json_block is not None:
                parsed = output_schema.model_validate(json_block)
            else:
                parsed = output_schema.model_validate_json("".join(text_parts))
            self.tokens.record("azure", model_name, in_tokens, out_tokens)
            return parsed, in_tokens, out_tokens

        content_text = "".join(text_parts).strip()
        self.tokens.record("azure", model_name, in_tokens, out_tokens)
        return content_text, in_tokens, out_tokens

    # ---------------------------------------------------------------- misc
    def _classify_error(self, exc: Exception) -> str:
        message = str(exc).lower()
        if "unauthorized" in message or "forbidden" in message or "401" in message or "403" in message:
            return "auth"
        if "invalid" in message or "bad request" in message or "schema" in message:
            return "invalid"
        if "rate" in message or "429" in message:
            return "rate"
        if "timeout" in message or "connection" in message or "502" in message or "504" in message:
            return "transient"
        return "transient"


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _convert_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content_list = content
        else:
            content_list = [{"type": "input_text", "text": str(content)}]
        converted.append({"role": msg.get("role", "user"), "content": content_list})
    return converted


__all__ = ["UnifiedLLM", "load_config", "TokenTracker", "RateLimiter", "ModelPool"]
