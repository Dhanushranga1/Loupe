# Setup guide

Loupe runs entirely locally — no API keys, no accounts. The only real decision is which **compute profile** to use, and that comes down to one question: do you have a GPU?

| Profile | Embedding model | Reranker | Use if... |
|---|---|---|---|
| `cpu_small` (default) | bge-small-en-v1.5 | MiniLM-L-6-v2 | You're on a laptop / CPU-only machine. Smallest, fastest to load, good enough for most repos. |
| `cpu_medium` | bge-base-en-v1.5 | MiniLM-L-6-v2 | CPU-only but you want somewhat better retrieval quality and don't mind slower indexing. |
| `gpu_large` | bge-large-en-v1.5 | bge-reranker-large | You have a CUDA GPU. Best quality, and the GPU makes the bigger models fast instead of slow. |

`loupe init` auto-detects a GPU for you and asks before switching to `gpu_large` — you don't have to know your hardware in advance.

## 1. Install (same for everyone)

```bash
git clone https://github.com/Dhanushranga1/Loupe.git
cd Loupe

cd core        && python -m venv .venv && .venv/bin/pip install -e .
cd ../mcp_server && python -m venv .venv && .venv/bin/pip install -e ../core -e .
cd ../cli        && python -m venv .venv && .venv/bin/pip install -e ../core -e ../mcp_server -e .
cd ..
```

GPU users: install a CUDA-enabled build of `torch` in each `.venv` *before* the steps above if you want `torch.cuda.is_available()` to actually find your GPU (the plain `pip install` above pulls a CPU-only wheel by default). See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command for your CUDA version.

## 2. If you have a GPU

```bash
cd /path/to/your/project
/path/to/Loupe/cli/.venv/bin/loupe init
```

Loupe detects the GPU and asks:

```
A GPU was detected. Use the higher-quality gpu_large profile instead of the default cpu_small? [y/N]
```

Answer `y`. Or skip the prompt entirely:

```bash
/path/to/Loupe/cli/.venv/bin/loupe init --compute-profile gpu_large
```

## 3. If you only have a CPU

Just run init — no GPU means no prompt, `cpu_small` is used automatically:

```bash
cd /path/to/your/project
/path/to/Loupe/cli/.venv/bin/loupe init
```

Want the better `cpu_medium` models anyway (slower index, better retrieval)?

```bash
/path/to/Loupe/cli/.venv/bin/loupe init --compute-profile cpu_medium
```

## 4. Index and serve (same for everyone, after init)

```bash
/path/to/Loupe/cli/.venv/bin/loupe index     # builds the symbol graph + embeddings
/path/to/Loupe/cli/.venv/bin/loupe serve     # starts the MCP server on :8765
```

## 5. Connect Claude Code

```bash
claude mcp add --transport http loupe http://127.0.0.1:8765/mcp
```

That's it — `search_symbols`, `get_symbol`, and the rest of the [MCP surface](../README.md#mcp-surface) are now available in that project.

## Notes

- The profile is committed in `loupe.manifest.yaml`, not a per-machine setting — everyone working on a shared repo should use the same one, since embedding dimensions differ between profiles and switching triggers a full reindex.
- Changing `compute_profile` later (editing the manifest, or re-running `loupe init --compute-profile ...`) triggers a full reindex automatically on the next `loupe index`/`loupe serve` — this is expected, not a bug.
- You can override just the embedding or reranker model without adopting a whole profile by setting `embedding_model:`/`cross_encoder_model:` directly in `loupe.manifest.yaml` (defaults to `auto`, meaning "whatever the profile says").
