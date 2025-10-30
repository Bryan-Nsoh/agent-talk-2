import asyncio
import time
from pathlib import Path

import pytest

from bioqueryous.llm_clients.unified_llm import ModelPool, RateLimiter, PoolsConfig, load_config


def test_load_config_from_models_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(Path.cwd())  # ensure relative search sees repo copy
    cfg, limits = load_config()
    assert "azure" in cfg.providers.raw
    pool = cfg.pools.raw.get("gpt-5-mini")
    assert pool and len(pool) == 2
    assert limits["providers"]["azure"]["tpm"] == 2700000


def test_model_pool_round_robin():
    async def runner():
        pools = {
            "toy": [
                {"provider": "azure", "deployment": "model-a"},
                {"provider": "azure", "deployment": "model-b"},
            ]
        }
        pool = ModelPool(PoolsConfig(raw=pools))
        provider1, entry1 = await pool.select("toy")
        provider2, entry2 = await pool.select("toy")
        assert provider1 == provider2 == "azure"
        assert entry1["deployment"] != entry2["deployment"]

    asyncio.run(runner())


def test_rate_limiter_enforces_concurrency():
    limits = {
        "defaults": {"concurrency": 1, "rpm": 0, "tpm": 0},
        "providers": {},
        "models": {},
    }
    limiter = RateLimiter(limits)

    async def runner():
        async def hold(name: str, events: list[str]):
            release = await limiter.acquire("azure", name, estimated_tokens=0)
            events.append(f"start-{name}")
            await asyncio.sleep(0.1)
            await release(None)
            events.append(f"end-{name}")

        events: list[str] = []
        await asyncio.gather(hold("first", events), hold("second", events))
        assert events[:2] == ["start-first", "end-first"]

    asyncio.run(runner())
