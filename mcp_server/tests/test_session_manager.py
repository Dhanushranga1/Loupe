"""Tests for app/session_manager.py (docs/phase-4-systems.md §8 — Session isolation)."""

from app.session_manager import SessionManager


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_two_sessions_are_fully_independent():
    manager = SessionManager()
    a = manager.get_or_create("session-a")
    b = manager.get_or_create("session-b")

    a.token_used = 5000
    assert b.token_used == 0, "one session's state must never leak into another"

    # re-fetching returns the same underlying object each time
    assert manager.get_or_create("session-a") is a
    assert manager.get_or_create("session-b") is b


def test_session_idle_past_ttl_is_swept():
    clock = FakeClock()
    manager = SessionManager(ttl_seconds=100, clock=clock)
    manager.get_or_create("stale")

    clock.advance(101)
    expired = manager.sweep_expired()

    assert expired == ["stale"]
    assert "stale" not in manager
    assert len(manager) == 0


def test_active_session_is_not_swept():
    clock = FakeClock()
    manager = SessionManager(ttl_seconds=100, clock=clock)
    manager.get_or_create("active")

    clock.advance(50)
    manager.get_or_create("active")  # touched again, refreshes last_active
    clock.advance(60)  # total elapsed since creation = 110 > ttl, but only 60 since last touch

    expired = manager.sweep_expired()
    assert expired == []
    assert "active" in manager


def test_sweep_only_removes_expired_not_all_sessions():
    clock = FakeClock()
    manager = SessionManager(ttl_seconds=100, clock=clock)
    manager.get_or_create("old")
    clock.advance(150)
    manager.get_or_create("new")
    clock.advance(20)  # old is now 170s idle, new is 20s idle

    expired = manager.sweep_expired()
    assert expired == ["old"]
    assert "new" in manager
    assert "old" not in manager
