"""Tests for retrieval/hyde.py (docs/PhaseX/experimental-gate-and-hyde.md, Part 2).

Uses the real local embedding model (free, no credentials needed) for the
embed-and-search half of the algorithm — matching this project's own
"real embeddings, not mocked" convention (see test_semantic.py). Only the
*generation* step is faked, via `FakeLLMClient`, deliberately: that's the
one real, paid, credentialed call this project has decided not to wire in
without explicit go-ahead. `FakeLLMClient` never imports or calls any real
LLM SDK — these tests make zero network calls of any kind.
"""

from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from loupe_core.parsing.schema import Symbol, SymbolKind
from loupe_core.retrieval.hyde import HYDE_PROMPT_TEMPLATE, LLMResponse, hyde_search
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME, SemanticIndex


class FakeLLMClient:
    """Records every prompt it's called with and returns a canned response — no network call."""

    def __init__(self, response_text: str, total_tokens: int = 42) -> None:
        self._response_text = response_text
        self._total_tokens = total_tokens
        self.prompts_seen: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str) -> LLMResponse:
        self.call_count += 1
        self.prompts_seen.append(prompt)
        return LLMResponse(text=self._response_text, total_tokens=self._total_tokens)


@pytest.fixture(scope="module")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture
def indexed_symbols(real_model):
    target = Symbol(
        id="a" * 16, kind=SymbolKind.FUNCTION, name="retry_with_backoff", qualified_name="retry_with_backoff",
        file_path="net.py", byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def retry_with_backoff(fn):",
        docstring="Retries a failed network call with exponential backoff, giving up after the maximum attempts.",
    )
    decoy = Symbol(
        id="b" * 16, kind=SymbolKind.FUNCTION, name="clear_cache", qualified_name="clear_cache",
        file_path="cache.py", byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def clear_cache():",
        docstring="Empty the in-memory cache of previously computed results.",
    )
    index = SemanticIndex(model=real_model)
    index.index([target, decoy])
    return index, target, decoy


def test_hyde_search_calls_the_llm_client_exactly_once_with_the_query_embedded_in_the_prompt(indexed_symbols):
    index, target, _decoy = indexed_symbols
    client = FakeLLMClient(response_text=target.docstring)

    hyde_search("how do we handle a flaky connection", client, index, top_k=5)

    assert client.call_count == 1
    assert client.prompts_seen[0] == HYDE_PROMPT_TEMPLATE.format(query="how do we handle a flaky connection")


def test_hyde_search_ranks_by_the_hypothetical_text_not_the_raw_query(indexed_symbols):
    """The whole point of HyDE: search runs against the *generated* text, so a
    fake client returning text close to `target` must surface `target` first,
    even though nothing about the raw call site mentions it directly."""
    index, target, decoy = indexed_symbols
    client = FakeLLMClient(response_text="def retry_with_backoff(fn):\n    " + target.docstring)

    result = hyde_search("some vague unrelated-sounding query", client, index, top_k=5)

    ranked_ids = [sid for sid, _ in result.ranked]
    assert ranked_ids[0] == target.id
    assert result.hypothetical_text == client._response_text


def test_hyde_search_propagates_total_tokens_from_the_llm_response(indexed_symbols):
    index, target, _decoy = indexed_symbols
    client = FakeLLMClient(response_text=target.docstring, total_tokens=137)

    result = hyde_search("query", client, index, top_k=5)

    assert result.total_tokens == 137
