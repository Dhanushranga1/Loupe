"""HyDE — Hypothetical Document Embeddings (docs/PhaseX/experimental-gate-and-hyde.md, Part 2).

One call to a generative LLM writes a plausible hypothetical answer for a
query; that hypothetical text is embedded with the same local embedding
model already loaded for retrieval (free — only the generation step costs
real tokens), then used as an extra semantic query whose ranked results
feed into RRF fusion as an independent fourth signal
(`fusion.fuse()`'s `hyde_results` parameter) — never a replacement for the
existing lexical/semantic/centrality pipeline, so HyDE can only ever add a
candidate signal, never remove one; it can't make retrieval worse than the
non-HyDE baseline.

`llm_client` is always injected, matching this project's established
dependency-injection pattern (e.g. `SemanticIndex(model=...)`). This module
imports no real LLM SDK and makes no network call itself — production
wiring of a *real*, credentialed client is a deliberate, separate decision
left to whoever operates a Loupe server with `hyde_query_rewrite` enabled;
nothing here assumes or requires one to exist. As of this writing, no such
client is constructed anywhere in this project's own server startup, so
HyDE stays fully inert (zero LLM calls, however the manifest is configured)
until an operator wires one in themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from loupe_core.retrieval.semantic import SemanticIndex

HYDE_PROMPT_TEMPLATE = (
    "Write a short, plausible, hypothetical answer or code snippet that "
    "would appear in this codebase in response to the following query.\n\n"
    "Query: {query}"
)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    total_tokens: int


class LLMClient(Protocol):
    def generate(self, prompt: str) -> LLMResponse: ...


@dataclass(frozen=True)
class HydeResult:
    ranked: list[tuple[str, float]]
    hypothetical_text: str
    total_tokens: int


def hyde_search(query: str, llm_client: LLMClient, semantic_index: SemanticIndex, top_k: int) -> HydeResult:
    """§6's algorithm: one generation call, then a free local-embedding search
    against the hypothetical text — same shape as `semantic_index.query()`,
    just against generated text instead of the raw query.
    """
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    response = llm_client.generate(prompt)
    ranked = semantic_index.query(response.text, top_k=top_k)
    return HydeResult(ranked=ranked, hypothetical_text=response.text, total_tokens=response.total_tokens)
