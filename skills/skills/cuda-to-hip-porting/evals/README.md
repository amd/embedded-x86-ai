<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
See LICENSE for license information.
-->

# Behavioral Evals — cuda-to-hip-porting

This folder contains the behavioral test suite for the `cuda-to-hip-porting` skill.
These tests are executed by the AMD Skills CI pipeline at **Stage 3: Agentic Automations**
and can also be run locally before submitting a PR.

---

## Purpose

The evals verify that the skill behaves correctly across three dimensions:

1. **Trigger correctness** — the skill activates on the right prompts and stays silent on unrelated ones
2. **Action correctness** — the skill executes the expected sequence of steps (hipify, proactive fixes, GPU detection, build)
3. **Artifact correctness** — the skill produces the expected output files and log entries

These tests run a **real agent session** against the skill — not mocks. The agent is
given the skill and a prompt, and the harness inspects what it actually did.

> **10-minute CI cap:** Because the full porting workload (hipify + build + run) can
> exceed the CI time limit, evals assert only the high-signal parts — correct activation
> and opening actions. Full end-to-end validation against real hardware is owned by the
> product repo.

---

## Files

| File | Description |
|------|-------------|
| `evals.py` | All behavioral test cases for this skill |
| `README.md` | This file — purpose, phase descriptions, check types, CI context |

---

## Stage 3 test items (from AMD Skills ROCm 10 intake spec)

The AMD Skills pipeline enforces these items via the `behavioral` workflow. All are
**blocking** (PR cannot merge if any fail), except Security scans which are advisory.

| Test item | What it checks | How enforced | Blocking? |
|-----------|---------------|-------------|-----------|
| Static prompt/action response functional test | Skill activates in a fixed agentic workspace and runs its expected actions | `behavioral` harness | Yes |
| Artifact & log generation screening | Expected artifacts/logs are produced | `workspace_contains`, `logs_contains` | Yes |
| False-positive execution | Skill is NOT triggered by non-trigger prompts | `should_not` | Yes |
| False-negative execution | Skill IS triggered by trigger prompts | `should` | Yes |
| Positive prompt/action screening | Correct activation and the expected sequence of actions | `behavioral` harness | Yes |
| Negative prompt/action screening | Correct activation without unintended actions | `behavioral` harness | Yes |
| Security scans | SkillSpector report reviewed by skill owner and findings triaged | `skillspector` workflow | Advisory |

---

## Test functions and which items they satisfy

### `test_port_vectoradd_to_hip` — Static functional + Artifact + Positive screening

The golden-path test: a real CUDA file is presented, the skill must hipify it, apply
proactive fixes (S67–S71), detect the GPU architecture, and build with `hipcc`.

**Deterministic checks** (`logs_contains`, `workspace_contains`):
- `logs_contains("hipify-perl")` — hipify was actually invoked
- `logs_contains("amdgpu-arch")` — GPU detection ran
- `logs_contains("Proactive HIP Fixes")` — S67–S71 fix step ran
- `workspace_contains("*.hip")` — a HIP output file was produced
- `workspace_contains("proactive_fixes.sh")` — fix script was generated

**LLM-judged `should`** (positive action screening):
- Run hipify-perl to convert `.cu` to `.hip`
- Apply proactive fixes for wave64 and cooperative groups
- Detect GPU architecture using `amdgpu-arch`
- Build with `hipcc --offload-arch`

**LLM-judged `should_not`** (negative action screening):
- Leave CUDA-specific includes like `cuda_runtime.h` unchanged
- Skip the proactive fix step after hipify

---

### `test_port_cublas_to_hipblas` — Library mapping (S11, S26)

Verifies the skill correctly maps cuBLAS headers, handle types, and build flags to
their hipBLAS equivalents when a library-dependent codebase is presented.

**LLM-judged `should`**: header swap, type rename, `-lhipblas` link flag

**LLM-judged `should_not`**: skill must not suggest keeping cuBLAS on AMD hardware

---

### `test_wave64_warp_fix` — Wave64 warp-size fix (S67)

Tests that hardcoded warp size `32` is replaced with `warpSize`. Critical for AMD CDNA
GPUs (gfx90a, gfx940) where `warpSize = 64` — leaving `32` causes silent numerical errors.

**LLM-judged `should`**: replace `32` with `warpSize`, explain wave64 implications

**LLM-judged `should_not`**: leave the hardcoded value unchanged

---

### `test_not_triggered_unrelated_prompt` — False-positive guard

Tests that the skill does **not** activate on a completely unrelated prompt (Python
generators). Prevents porting context from being injected into unrelated sessions.

**LLM-judged `should_not`**: no HIP/ROCm content appears in the response

---

### `test_not_triggered_nvidia_only` — NVIDIA-only false-positive guard

Tests that the skill does **not** activate when the user is explicitly working with
NVIDIA hardware on a non-porting task.

**LLM-judged `should_not`**: skill must not suggest porting when user is on NVIDIA

---

## Check types

| Type | How evaluated | Blocking? |
|------|--------------|-----------|
| `logs_contains(string)` | Deterministic — exact string match in agent execution logs | Yes |
| `workspace_contains(pattern)` | Deterministic — glob pattern match in working directory | Yes |
| `run.should(description)` | LLM-judged — Claude grades whether the described action occurred | Yes |
| `run.should_not(description)` | LLM-judged — Claude grades whether the described action was absent | Yes |

---

## Running locally

```bash
# From the amd/skills repo root
cd eval/behavioral
python -m pytest -c pytest.ini -p conftest \
    ../../skills/cuda-to-hip-porting/evals/evals.py -v
```

Requirements:
- `claude` CLI installed and authenticated
- AMD ROCm GPU available (GPU-dependent checks skip if none detected)
- Skill installed: `cp -r skills/cuda-to-hip-porting ~/.claude/skills/`

---

## CI integration

These evals are enforced by the `behavioral` GitHub Actions workflow, which runs on
every PR push that touches `skills/cuda-to-hip-porting/`. The workflow executes on
self-hosted Strix Halo runners (Linux + Windows) inside a device-passthrough container.

The `skillspector` workflow runs in parallel and produces a security scan report.
Findings must be reviewed and triaged by the skill owner before merge (advisory, not blocking).

All behavioral test functions are **blocking gates** — the PR cannot merge if any fail.
