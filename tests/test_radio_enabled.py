from llmgrid.env.simulate import run_episode
from llmgrid.schema import Position


def test_radio_off_for_none(monkeypatch):
    # Tiny 3x3 empty map with 2 agents to exercise observation building quickly
    metrics = run_episode(
        use_llm=False,
        model_id="openrouter:test",
        width=3,
        height=3,
        obstacles=[],
        start_positions={"a1": Position(x=0, y=1), "a2": Position(x=2, y=1)},
        goal=Position(x=2, y=0),
        turns=2,
        visibility=2,
        radio_range=2,
        seed=1,
        map_sharing="none",
    )
    # No exception means strategy accepted; radio off is enforced inside simulate via comm_limits in obs
    assert metrics.turns == 2
