# Nik Lowerson

LLM inference on AMD GPUs — ROCm, HIP, vLLM. I care about measuring before claiming.

### About

I work close to the metal on inference: the runtime, the kernels, the memory layout, and
the numbers that prove any of it. Most of what I build starts as a question I could not
answer from a datasheet — "does this host even have a working ROCm stack", "how much VRAM
will 32k of context actually eat" — and ends as a small read-only tool with tests.

Certified by AMD on this stack in 2026. The certificate is below; the repository is the
part that can be checked.

### Focus

- Single-node serving on CDNA hardware (MI300X, `gfx942`), fp8 KV budgets that fit before the model loads.
- HIP porting from CUDA, and the gaps that porting exposes.
- Latency work — TTFT, ITL, p99 — with the counting method stated next to every number.

### Skills

`ROCm` `HIP` `CDNA / RDNA architecture` `rocprofv3 / roctx` `memory coalescing` `LDS tiling`
`FP16/BF16 GEMM with FP32 accumulation` `vLLM` `KV-cache budgeting` `Python, stdlib-first` `Linux`

### Certifications

[![AMD ROCm Certified Associate](https://images.credly.com/size/340x340/images/cc8d77e0-0fb9-4c29-b326-2cb9776bfd4c/blob)](https://www.credly.com/badges/bc3a84e3-1ce0-42f5-a900-b3d2171a9806)

**AMD ROCm Certified Associate** · AMD · issued Nov 2026, valid through Nov 2028

ROCm stack and HIP porting, CDNA/RDNA architecture, rocprofv3 / roctx profiling, memory
coalescing and LDS tiling, FP16/BF16 GEMM with FP32 accumulation.

### Google Developer Program

[Public developer profile →](https://me.developers.google.com/u/113698327822147498392)

Member of the Google Cloud & NVIDIA community. Passed the quiz of all five joint learning
paths on the Google Cloud + NVIDIA stack — cuDF/cuML acceleration without code rewrites,
GPU inference pipelines, NVIDIA NIM on GKE, JAX on GPU, GPU analytics:

| Path | Score |
|---|---|
| Accelerated ML with Google Cloud and NVIDIA | 9/9 |
| Intro to Inference: How to Run AI Models on a GPU | 5/5 |
| Deploy Faster GenAI models with NVIDIA NIM on GKE | 7/8 |
| Scale and Serve JAX Models on NVIDIA GPUs | 16/17 |
| Speed Up Data Analytics on GPUs | 7/7 |

### Projects

- [amd-llm-kit](https://github.com/wZ8ss88/amd-llm-kit) — ROCm host probe, gfx target table, KV-cache sizing, VRAM watcher, TTFT/ITL benchmark. Standard library only, 22 tests.

### How I work

- Read-only by default. Probes report `MISSING` on the wrong host instead of guessing.
- Numbers carry provenance — a token count derived from content deltas is tagged as an
  approximation, never dressed up as exact.
- If a claim cannot be tested here, the README says so rather than leaving it implied.
