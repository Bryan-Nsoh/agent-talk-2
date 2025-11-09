from llmgrid.prompts import build_prompt_header
from llmgrid.schema import resolve_strategy_capabilities


def test_prompt_excludes_comm_sections_for_none_strategy():
    caps = resolve_strategy_capabilities("none", oracle_enabled=False)
    header = build_prompt_header(
        radio_range=0,
        action_kinds=caps.action_kinds,
    )

    assert "COMMUNICATE — one structured message" not in header
    assert "peer radio is disabled" in header.lower()
    assert "\"kind\":\"MOVE|STAY\"" in header


def test_prompt_mentions_comm_when_enabled():
    caps = resolve_strategy_capabilities("structured", oracle_enabled=False)
    header = build_prompt_header(
        radio_range=2,
        action_kinds=caps.action_kinds,
    )

    assert "COMMUNICATE — one structured message" in header
    assert "range 2" in header
    assert "\"kind\":\"MOVE|STAY|COMMUNICATE\"" in header
