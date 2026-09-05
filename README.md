# RouteCache — Qwen3.8-27B UD-IQ2_S

Clean public runtime, GGUF-embedded RouteCache execution profile, one-click Windows installer, direct Ollama support, and optional full model reproduction.

## One-click RouteCache

```bat
SETUP_AND_RUN.bat
```

It auto-installs Python through WinGet if needed, downloads the published GGUF from Hugging Face, detects the GPU, attempts the pinned custom CUDA runtime when build tools are already present, otherwise downloads the known-good GitHub Release fallback, pre-warms the model, enables low-latency streaming and automatic 4096-token rolling context compaction, and opens chat.

## One-click Ollama

```bat
OLLAMA_SETUP_AND_RUN.bat
```

If Ollama is missing, Windows setup attempts `winget install --id Ollama.Ollama -e`, then runs the GGUF directly from the configured Hugging Face repo. The HF repo root `params` file sets `num_ctx=4096`; that is what the HF→Ollama bridge reads.

Manual equivalent:

```text
ollama run hf.co/CyberGamer/Qwen3.8-27B-UD-IQ2_S-RouteCache-GGUF
```

Ollama uses the same GGUF/tokenizer/context setting but does not reproduce RouteCache's explicit llama.cpp `-ot` placement/custom runtime, so the RouteCache TPS calibration does not apply to Ollama.

## Full reproduction from official Qwen weights

```bat
REPRODUCE_MODEL.bat
```

Or use `REPRODUCE_AND_RUN.bat` for the complete advanced path: rebuild → install runtime → open chat.

This advanced path needs about 120–140 GiB temporary free space. It downloads official `Qwen/Qwen3.8-27B` BF16 weights, pinned llama.cpp source, Unsloth's public `imatrix_unsloth.gguf`, reads the public UD-IQ2_S GGUF header by HTTP range requests to capture the per-tensor dynamic quantization map, converts the official weights to BF16 GGUF, prunes the MTP block and rewrites `qwen35.block_count=64` / `qwen35.nextn_predict_layers=0`, quantizes with that published recipe, and verifies the resulting tensor topology/types.

Normal users should **not** rebuild the model; they should download the published 8.37 GB GGUF.

## Certified profile

The explicit measured route is only selected on the matching RTX 4060 8 GB profile: 4096 context, Q8_0 KV, batch/ubatch 256/64, token embedding on CPU, FFN up/gate blocks 33–46 on CPU, fitter off. The original warmed non-MTP floor measured about 11.39 tok/s. Other hardware uses portable auto-fit.

## Other commands

- `INSTALL.bat` — install only
- `RUN_CHAT.bat` — browser chat
- `RUN_API.bat` — API only
- `REBUILD_KERNEL.bat` — retry custom GPU runtime
- `CREATE_OLLAMA.bat` — create local Ollama model after GGUF download
- `DIAGNOSTICS.bat` — environment/runtime state


## Embedded RouteCache metadata

The GGUF contains namespaced `routecache.*` metadata including the certified RTX 4060 tensor route, context/batch/ubatch/KV settings, warm-up/context-compaction policy, source GGUF SHA-256 and calibration metadata. The RouteCache runtime reads `routecache.profile.json` from the GGUF at launch. Stock llama.cpp/Ollama may ignore these custom keys and load the same tensors normally.
