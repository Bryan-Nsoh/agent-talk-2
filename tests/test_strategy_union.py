from llmgrid.schema import build_decision_model
from pydantic import ValidationError


def validates(model, action: dict) -> bool:
    try:
        model.model_validate({"action": action})
        return True
    except ValidationError:
        return False


def test_none_rejects_communicate_allows_move_stay_mark():
    m = build_decision_model("none", oracle_enabled=False)
    assert validates(m, {"kind": "MOVE", "direction": "N", "payload": None})
    assert validates(m, {"kind": "STAY", "payload": None})
    assert not validates(m, {"kind": "COMMUNICATE", "message": {"kind": "CHAT", "sender_id": "a1", "seq": 1, "text": "hi"}})


def test_structured_accepts_communicate_with_request_or_intent():
    m = build_decision_model("structured", oracle_enabled=False)
    # INTENT
    assert validates(m, {"kind": "COMMUNICATE", "message": {"kind": "INTENT", "sender_id": "a1", "seq": 1, "next_action": "MOVE_N"}})
    # REQUEST
    assert validates(m, {"kind": "COMMUNICATE", "message": {"kind": "REQUEST", "sender_id": "a1", "seq": 2, "req": "YIELD", "target": {"x": 5, "y": 5}}})


def test_freeform_accepts_chat_only():
    m = build_decision_model("freeform", oracle_enabled=False)
    assert validates(m, {"kind": "COMMUNICATE", "message": {"kind": "CHAT", "sender_id": "a1", "seq": 1, "text": "REROUTE north"}})
    # Structured messages should be rejected in freeform
    assert not validates(m, {"kind": "COMMUNICATE", "message": {"kind": "INTENT", "sender_id": "a1", "seq": 1, "next_action": "MOVE_N"}})
