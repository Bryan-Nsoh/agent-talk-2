from llmgrid.schema import Decision, MoveAction, Direction


def test_decision_serialises_roundtrip():
    decision = Decision(action=MoveAction(direction=Direction.N))
    payload = decision.model_dump(mode="json")
    restored = Decision.model_validate(payload)
    assert restored.action.kind == "MOVE"
    assert restored.action.direction == Direction.N
