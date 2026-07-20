"""Tests for context/session_notes.py (docs/PhaseX/session-notes.md).

Uses the real embedding model (session-scoped) for the pipeline/MMR tests —
same "real model, not fabricated data" discipline as test_fusion.py/test_rerank.py.
"""

import pytest
from sentence_transformers import SentenceTransformer

from loupe_core.context.session_notes import DEFAULT_ACTIVE_SET_LIMIT, SessionNotesStore
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME


@pytest.fixture(scope="session")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture
def store(tmp_path, real_model):
    return SessionNotesStore("session-1", logs_dir=tmp_path, model=real_model)


# --------------------------------------------------------------------------
# §6 acceptance criterion: decayed importance beats pure recency
# --------------------------------------------------------------------------


def test_high_importance_early_note_outranks_low_importance_recent_note(store):
    """Constructed so pure recency gives the wrong answer: the early note has
    much higher importance and only a few turns have passed, so its decayed
    score still exceeds the fresh, low-importance note's score — verified
    against the real EvictionCache decay math, not assumed.
    """
    early = store.write("The root cause is in token refresh, not login.", importance=5)
    store.write("checked file X, looks fine", importance=1)
    store.write("checked file Y, also fine", importance=1)
    recent = store.write("noted the timeout config value", importance=1)

    early_score = store._eviction.current_scores[early.note_id]
    recent_score = store._eviction.current_scores[recent.note_id]

    assert early_score > recent_score, (
        "a high-importance note from a few turns ago must still outrank a low-importance fresh one"
    )
    # Sanity check that this isn't vacuous: pure recency (last-written-first)
    # would rank `recent` above `early`, the opposite conclusion.
    assert recent.turn_index > early.turn_index


# --------------------------------------------------------------------------
# §6 acceptance criterion: active-set eviction vs. permanent full-log presence
# --------------------------------------------------------------------------


def test_active_set_eviction_does_not_delete_from_the_full_log(tmp_path, real_model):
    store = SessionNotesStore("session-2", logs_dir=tmp_path, active_set_limit=3, model=real_model)

    first = store.write("first note, low importance", importance=1)
    store.write("second note", importance=3)
    store.write("third note", importance=3)
    store.write("fourth note pushes the active set over its limit", importance=3)

    assert not store.is_active(first.note_id), "the lowest-decayed-importance note must be evicted from the active set"
    assert len(store.read_recent(limit=100)) == 3, "active set must not exceed its configured limit"

    full_log_ids = {n.note_id for n in store.list_all()}
    assert first.note_id in full_log_ids, "an evicted note must still be present in the full append-only log"
    assert len(store.list_all()) == 4, "the full log must contain every note ever written, eviction or not"


def test_default_active_set_limit_is_a_documented_constant():
    assert DEFAULT_ACTIVE_SET_LIMIT == 50


def test_write_rejects_importance_outside_1_to_5(store):
    with pytest.raises(ValueError):
        store.write("bad importance", importance=0)
    with pytest.raises(ValueError):
        store.write("bad importance", importance=6)


# --------------------------------------------------------------------------
# §6 acceptance criterion: MMR deduplication of near-duplicate notes
# --------------------------------------------------------------------------


def test_read_relevant_deduplicates_near_duplicate_notes_via_mmr(store):
    store.write("token refresh looks broken", importance=4)
    store.write("confirmed token refresh is the issue", importance=4)
    store.write("token refresh bug is in the retry decorator", importance=4)
    store.write("the CSS on the login page is misaligned", importance=3)
    store.write("database connection pool size might need tuning", importance=3)

    results = store.read_relevant("what's wrong with token refresh", top_k=3)
    contents = [n.content for n in results]

    assert len(results) == 3
    # At least one genuinely distinct note (not about token refresh at all)
    # must survive selection rather than all 3 slots going to near-duplicates.
    distinct_present = any("CSS" in c or "database" in c for c in contents)
    assert distinct_present, f"MMR must not exhaust every slot on near-duplicate token-refresh notes, got: {contents}"


# --------------------------------------------------------------------------
# §6 acceptance criterion: full pipeline end-to-end on a >=10-note session
# --------------------------------------------------------------------------


def test_full_read_relevant_pipeline_on_a_ten_note_session(tmp_path, real_model):
    store = SessionNotesStore("session-3", logs_dir=tmp_path, model=real_model)

    store.write("auth middleware validates JWT tokens on every request", importance=5)
    store.write("JWT validation logic lives in middleware/auth.py", importance=4)
    store.write("checked the login form styling, looks fine", importance=1)
    store.write("database migrations run automatically on deploy", importance=2)
    store.write("the retry decorator has an off-by-one bug", importance=5)
    store.write("off-by-one bug is in retry_decorator.py line 42", importance=4)
    store.write("verified the CSS grid layout on the dashboard", importance=1)
    store.write("payment webhook signature verification uses HMAC", importance=3)
    store.write("logging format was recently changed to structured JSON", importance=2)
    store.write("token refresh interval is configurable via env var", importance=3)

    results = store.read_relevant("what's the bug in the retry logic", top_k=3)

    assert 1 <= len(results) <= 3
    top_contents = " ".join(n.content for n in results)
    assert "retry" in top_contents.lower() or "off-by-one" in top_contents.lower()


# --------------------------------------------------------------------------
# §6 acceptance criterion: compaction survival
# --------------------------------------------------------------------------


def test_note_written_mid_session_survives_many_subsequent_turns(store):
    """Notes live server-side, independent of Claude Code's own conversation
    transcript — writing many further notes (simulating many more turns/a
    conversation compaction event happening at the transcript level) must
    never affect an earlier note's retrievability from the same store.
    """
    early = store.write("the root cause is in token refresh, not login", importance=5)
    for i in range(15):
        store.write(f"filler note {i}", importance=1)

    full_log_ids = {n.note_id for n in store.list_all()}
    assert early.note_id in full_log_ids


def test_a_fresh_store_instance_sees_the_full_log_written_by_an_earlier_instance(tmp_path, real_model):
    """The stronger, literal proof of persistence: notes survive independent
    of any single Python object's lifetime, not just within one still-running
    process — a fresh `SessionNotesStore` pointed at the same session_id and
    logs_dir sees everything a prior instance wrote.
    """
    first_instance = SessionNotesStore("session-4", logs_dir=tmp_path, model=real_model)
    first_instance.write("the root cause is in token refresh, not login", importance=5)

    second_instance = SessionNotesStore("session-4", logs_dir=tmp_path, model=real_model)
    contents = [n.content for n in second_instance.list_all()]

    assert "the root cause is in token refresh, not login" in contents


# --------------------------------------------------------------------------
# Session isolation
# --------------------------------------------------------------------------


def test_two_different_sessions_notes_never_cross_contaminate(tmp_path, real_model):
    store_a = SessionNotesStore("session-a", logs_dir=tmp_path, model=real_model)
    store_b = SessionNotesStore("session-b", logs_dir=tmp_path, model=real_model)

    store_a.write("session A's private note", importance=3)
    store_b.write("session B's private note", importance=3)

    a_contents = {n.content for n in store_a.list_all()}
    b_contents = {n.content for n in store_b.list_all()}

    assert a_contents == {"session A's private note"}
    assert b_contents == {"session B's private note"}
