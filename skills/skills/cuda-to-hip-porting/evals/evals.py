# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `cuda-to-hip-porting` skill.

Run locally (needs the `claude` CLI authenticated and an AMD ROCm GPU available
-- otherwise GPU-dependent checks are skipped):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest ../../skills/cuda-to-hip-porting/evals/evals.py

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure, so
the test fails at the first unmet expectation. `logs_contains` /
`workspace_contains` are deterministic; `should` / `should_not` are graded by
an LLM judge over the captured evidence.
"""

from harness import claude


def test_port_vectoradd_to_hip():
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="cuda-to-hip-porting") as agent:
            run = agent.prompt(
                "Port this vectorAdd.cu file to AMD HIP for ROCm. "
                "Run hipify-perl, apply proactive fixes, and build it."
            )

            # Deterministic expectations
            run.logs_contains("hipify-perl")
            run.logs_contains("amdgpu-arch")
            run.logs_contains("Proactive HIP Fixes")
            run.workspace_contains("*.hip")
            run.workspace_contains("proactive_fixes.sh")

            # Positive behavioral expectations
            run.should("Run hipify-perl to convert the .cu file to .hip")
            run.should("Apply proactive fixes for wave64 and cooperative groups")
            run.should("Detect the GPU architecture using amdgpu-arch")
            run.should("Build the ported code with hipcc --offload-arch")

            # Negative behavioral expectations
            run.should_not("Leave CUDA-specific includes like cuda_runtime.h unchanged")
            run.should_not("Skip the proactive fix step after hipify")


def test_port_cublas_to_hipblas():
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="cuda-to-hip-porting") as agent:
            run = agent.prompt(
                "Help me port my cuBLAS code to hipBLAS on ROCm."
            )

            run.should("Replace cublas headers with hipblas/hipblas.h")
            run.should("Rename cublasHandle_t to hipblasHandle_t")
            run.should("Update the build flags to link with -lhipblas")
            run.should_not("Suggest using cuBLAS on an AMD GPU")


def test_wave64_warp_fix():
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="cuda-to-hip-porting") as agent:
            run = agent.prompt(
                "My warp-reduction code has hardcoded 32 — fix it for AMD wave64."
            )

            run.should("Replace hardcoded warp size 32 with warpSize")
            run.should("Explain that AMD CDNA GPUs use wave64 where warpSize equals 64")
            run.should_not("Leave the hardcoded value of 32 unchanged")


def test_not_triggered_unrelated_prompt():
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="cuda-to-hip-porting") as agent:
            run = agent.prompt(
                "Explain Python generators and the yield keyword."
            )

            run.should_not("Run hipify-perl or mention HIP porting")
            run.should_not("Reference AMD GPU architecture or ROCm")


def test_not_triggered_nvidia_only():
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="cuda-to-hip-porting") as agent:
            run = agent.prompt(
                "How do I train a PyTorch model on a single NVIDIA GPU?"
            )

            run.should_not("Suggest porting to HIP or AMD ROCm")
            run.should_not("Invoke the cuda-to-hip-porting skill")
