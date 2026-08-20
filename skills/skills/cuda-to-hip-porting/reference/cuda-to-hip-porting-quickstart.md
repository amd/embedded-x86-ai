<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# cuda-to-hip-porting: Quickstart — Your First Port in 10 Minutes

This walks through porting a simple CUDA project (like `vectorAdd`) to HIP using the
`cuda-to-hip-porting` Claude Code skill.

## Prerequisites

- `claude` (Claude Code) installed and authenticated.
- `nvcc` available (or a CUDA Docker image — see `skills/cuda-to-hip-porting/reference/docker-setup.md`).
- `hipcc` available (or a ROCm Docker image — same file).
- The `cuda-to-hip-porting` skill installed to `~/.claude/skills/` (see the top-level README).
- **For gfx1151 (Strix Halo):** ROCm 7.2+ required — earlier versions crash at runtime.
- A CUDA project to port (your own, or download NVIDIA CUDA Samples from https://github.com/NVIDIA/cuda-samples)

## 1. Profile first

```bash
python3 skills/cuda-to-hip-porting/scripts/profile_cuda_project.py /path/to/your/cuda/project
```

You'll see which NVIDIA libraries, CUDA features, and tech-plan gaps are present.
For a simple project like `vectorAdd`, expect: 1 CUDA file, no libraries, zero gaps.

## 2. Ask Claude Code to do the port

From your CUDA project directory, start `claude` and paste a prompt from
`reference/cuda-to-hip-porting-example-prompts.md` (use Prompt A for a first port).

**For simple single-file projects** (like vectorAdd):
- Use Prompt A — standard workflow
- The skill will run hipify, apply manual rewrites, build, and verify

**For multi-file projects** (50+ files):
- Complete the CRITICAL SETUP CHECKLIST first (see `skills/cuda-to-hip-porting/SKILL.md`)
- Use Prompt H — includes parallelization and realistic expectations
- Process ALL files in parallel for 10–50× speedup
- Kernel-only files failing to build is NORMAL and EXPECTED

The skill will:

1. [Optional] Profile to understand scope
2. Complete setup checklist (hip_helper.h, Common/*.h, Docker permissions)
3. Run `hipify-perl` (parallel for large projects)
4. Read warnings and apply manual rewrites
5. Build with `hipcc` (realistic expectations: 5–10% standalone, 40–50% CMake)
6. Run and compare output against CUDA baseline

For simple projects, no manual rewrites are needed — it's pure hipify.

## 3. Verify

### Native (if you have both GPUs)

```bash
cd /path/to/your/cuda/project
nvcc yourfile.cu -o yourfile_cuda    # builds on NVIDIA host
hipcc yourfile.hip -o yourfile_hip   # builds on AMD host
mkdir -p ./tmp_output
./yourfile_cuda > ./tmp_output/out_cuda.txt
./yourfile_hip  > ./tmp_output/out_hip.txt
diff ./tmp_output/out_cuda.txt ./tmp_output/out_hip.txt && echo OK
```

### Docker (Auto-detect GPU)

**IMPORTANT**: Always use `--user $(id -u):$(id -g)` to avoid permission errors:

```bash
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete bash -c '
        GPU=$(amdgpu-arch | head -1)
        if [ -n "$GPU" ]; then
            hipcc -I. --offload-arch=$GPU -w -o vectorAdd_hip vectorAdd.hip && ./vectorAdd_hip
        else
            hipcc -I. -c -w -o vectorAdd.o vectorAdd.hip
            echo "Compile-only (no GPU detected)"
        fi
    '
```

**Note:** gfx1151 (Strix Halo) requires ROCm 7.2+. See `skills/cuda-to-hip-porting/reference/docker-setup.md` for details.

## 4. Move up the difficulty curve

Use NVIDIA CUDA Samples (https://github.com/NVIDIA/cuda-samples) to practice:

| Sample Type | Difficulty | Key Skills |
|-------------|------------|------------|
| vectorAdd, matrixMul | Trivial | S1, S2 (pure hipify) |
| cuBLAS samples | Easy | + S11, S18 (library linking, wave-size) |
| Cooperative groups | Medium | + S8, S10 (wave64 adaptation, PTX) |
| WMMA/Tensor Core | Hard | + S25 (hipify translates zero WMMA code) |

Start with simple runtime API samples, then try library samples. WMMA samples
require full manual rewrite — hipify produces no useful output for Tensor Core code.

## 5. Read what the skill knows

Open `skills/cuda-to-hip-porting/SKILL.md` and the files under `reference/`. The
recipes there are the actual knowledge you're transferring — the AI prompts just drive them.

| Reference File | Contents |
|----------------|----------|
| `api-rewrites.md` | S1–S56 skills, manual rewrite patterns |
| `troubleshooting.md` | 25+ error fixes |
| `docker-setup.md` | Container setup, gfx1151 ROCm 7.2 |
| `build-commands.md` | nvcc/hipcc flags, architecture mapping |
| `hipify-workflow.md` | hipify-perl vs hipify-clang |
