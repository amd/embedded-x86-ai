# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Gemma 4 E4B on the iGPU - the brain, via ``llama.cpp`` + ROCm/HIP.

The LLM does not run in-process: a ``llama-server`` (built with
``-DGGML_HIP=ON``) owns the iGPU and we talk to it over its OpenAI-compatible
HTTP API. Two things make this path AMD-specific, and both are your TODOs:

1. **Bringing the server up on the iGPU** (:func:`llama_server_command`):
   the launch flags and environment that keep a Strix APU happy -
   ``HSA_OVERRIDE_GFX_VERSION`` (the ROCm runtime needs the iGPU presented
   as a supported gfx target or the first kernel dispatch can segfault),
   ``--ctx-size`` capped at 2048 (the model's native 131072 KV cache OOMs
   the iGPU VRAM carve-out), ``--n-gpu-layers 99`` (everything on the GPU),
   and ``--no-warmup`` (gfx1151 can segfault in the warmup pass).
2. **Grammar-constrained intent parsing** (:func:`parse_intent`): sending a
   chat completion with the GBNF grammar attached, so the model physically
   cannot emit anything but ``{"command": ..., "object": ...}``.

The intent lesson is ``notebooks/02_ros2/03_intent_node.ipynb``.
Self-test: ``python -m models.selftest`` (uses a mock server - no GPU needed)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

import requests

from common.config import resolve
from common.intents import INTENT_SYSTEM_PROMPT, Intent, ParsedCommand

logger = logging.getLogger(__name__)


def llama_server_command(cfg: dict):
    """Build the ``llama-server`` launch: ``(cmd list, env dict)``.

    Reads ``gemma4-e4b-it:`` from the config. The command must include the model, host,
    port, ``--n-gpu-layers``, ``--ctx-size``, ``--parallel``, and
    ``--no-warmup`` when configured; the environment must carry
    ``HSA_OVERRIDE_GFX_VERSION`` when the config sets one (don't override a
    value the user already exported).
    """
    l = cfg["gemma4-e4b-it"]
    server_bin = resolve(l["server_bin"])
    model = resolve(l["model_gguf"])
    if not server_bin.exists():
        raise FileNotFoundError(f"{server_bin} not found - build llama.cpp "
                                "with -DGGML_HIP=ON first (see the repo README).")
    if not model.exists():
        raise FileNotFoundError(f"{model} not found - download the GGUF first.")

    # >>> TODO 1.5: llama_server_command
    cmd = [
        str(server_bin),
        "--model", str(model),
        "--host", l["host"],
        "--port", str(l["port"]),
        "--n-gpu-layers", str(l["n_gpu_layers"]),
        "--ctx-size", str(l["ctx_size"]),
        "--parallel", str(l.get("parallel", 1)),
    ]
    if l.get("mmproj"):
        cmd.extend(["--mmproj", str(resolve(l["mmproj"]))])
    if l.get("no_warmup", False):
        cmd.append("--no-warmup")

    env = os.environ.copy()
    gfx = l.get("hsa_override_gfx_version")
    if gfx:
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", str(gfx))
    return cmd, env
    # <<< TODO 1.5


def server_alive(base_url: str, http=None, timeout: float = 0.5) -> bool:
    """PROVIDED: ``GET /health`` → True on HTTP 200."""
    http = http or requests
    try:
        return http.get(f"{base_url}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def start_llama_server(cfg: dict, ready_timeout_s: float = 120.0,
                       log_dir: str = "logs") -> subprocess.Popen:
    """PROVIDED: spawn the server with YOUR command/env and wait for /health.

    Process plumbing is generic; the launch recipe (your TODO 1.5) is not.
    """
    cmd, env = llama_server_command(cfg)
    log_path = resolve(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / "llama-server.log", "w")
    logger.info("Starting llama-server (ROCm): %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            env=env)
    base = f"http://{cfg['gemma4-e4b-it']['host']}:{cfg['gemma4-e4b-it']['port']}"
    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if server_alive(base):
            logger.info("llama-server ready at %s", base)
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} during "
                f"startup - see {log_path / 'llama-server.log'}")
        time.sleep(0.5)
    proc.terminate()
    raise TimeoutError("llama-server did not become healthy in time.")


def parse_intent(transcript: str, base_url: str, grammar: str,
                 http=None, temperature: float = 0.0) -> ParsedCommand:
    """One transcript → one :class:`ParsedCommand`, grammar-constrained.

    POST an OpenAI-style chat completion to ``{base_url}/v1/chat/completions``
    with the system prompt (:data:`INTENT_SYSTEM_PROMPT`), the transcript as
    the user message, ``temperature`` (0 = deterministic routing), a small
    ``max_tokens``, and - the crucial part - the GBNF ``grammar``, which is
    what makes the output JSON *guaranteed parseable*. An empty transcript
    maps to UNKNOWN without any HTTP round trip.
    """
    http = http or requests
    # >>> TODO 1.6: parse_intent
    if not transcript or not transcript.strip():
        return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
    payload = {
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": transcript.strip()},
        ],
        "temperature": temperature,
        "max_tokens": 48,
        "grammar": grammar,
    }
    r = http.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        command = data["command"]
    except (json.JSONDecodeError, KeyError):
        logger.warning("Unparseable LLM output: %r", content)
        return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
    intent = Intent.from_string(command)
    params = {}
    obj = str(data.get("object", "") or "").strip()
    if obj:
        params["object"] = obj
    return ParsedCommand(intent, params=params, transcript=transcript)
    # <<< TODO 1.6


class LlamaIntentParser:
    """PROVIDED: lifecycle wrapper - connect or autostart, then parse.

    Keeps one HTTP session alive (saves ~5-20 ms per command) and owns the
    server process only if it spawned one.
    """

    def __init__(self, cfg: dict, autostart=None):
        l = cfg["gemma4-e4b-it"]
        self.base = f"http://{l['host']}:{l['port']}"
        self.temperature = float(l.get("temperature", 0.0))
        self.grammar = resolve(l["grammar_file"]).read_text(encoding="utf-8")
        self._http = requests.Session()
        self._proc = None
        autostart = l.get("autostart", True) if autostart is None else autostart
        if not server_alive(self.base, self._http):
            if autostart:
                self._proc = start_llama_server(cfg)
            else:
                raise ConnectionError(
                    f"llama-server not reachable at {self.base} and autostart "
                    "is disabled.")

    def parse(self, transcript: str) -> ParsedCommand:
        """Parse ``transcript`` via :func:`parse_intent`, logging latency."""
        t0 = time.perf_counter()
        cmd = parse_intent(transcript, self.base, self.grammar,
                           http=self._http, temperature=self.temperature)
        logger.info("Intent %r %s ← %r (%.0f ms)", cmd.intent.value,
                    cmd.params or "", transcript,
                    (time.perf_counter() - t0) * 1e3)
        return cmd

    def close(self) -> None:
        """Terminate the owned llama-server process, if this instance started one."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
