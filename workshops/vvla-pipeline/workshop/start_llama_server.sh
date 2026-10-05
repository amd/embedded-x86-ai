#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$REPO_ROOT/third_party/llama.cpp/build/bin/llama-server" \
    --model "$REPO_ROOT/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    --host 127.0.0.1 --port 8081 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --no-warmup
