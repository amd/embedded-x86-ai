# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Llama-3.2-3B intent parsing via llama.cpp running on ROCm.

The model runs in ``llama-server`` (built with ``-DGGML_HIP=ON`` by
``bootstrap.sh``) and is queried over its OpenAI-compatible HTTP API. A GBNF
grammar (``config/intent.gbnf``) constrains generation so the model can only
emit ``{"command": "<name>", "object": "<words>"}`` - no free-form parsing.

This revision returns a :class:`ParsedCommand` (intent **plus parameters**)
so commands like "pick up the ball" carry their object through to behaviors,
and keeps a persistent HTTP session (keep-alive) to shave connection setup
off the voice-command latency budget.

Independent test
----------------
    python -m vla_pipeline.llm.llama_intent --text "hey can you pick up the ball"
    python -m vla_pipeline.llm.llama_intent --selftest
    python -m vla_pipeline.llm.llama_intent           # interactive REPL
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time

import requests

from vla_pipeline.llm.intents import INTENT_SYSTEM_PROMPT, Intent, ParsedCommand
from vla_pipeline.utils.config import load_config, resolve

logger = logging.getLogger(__name__)


class LlamaIntentParser:
    """Manage a llama.cpp server and turn transcripts into :class:`ParsedCommand`."""

    def __init__(self, cfg: dict):
        l = cfg["llm"]
        self.host: str = l["host"]
        self.port: int = int(l["port"])
        self.base = f"http://{self.host}:{self.port}"
        self.temperature: float = float(l.get("temperature", 0.0))
        self.grammar = resolve(l["grammar_file"]).read_text(encoding="utf-8")
        self._cfg = l
        self._log_dir = cfg.get("system", {}).get("log_dir", "logs")
        self._proc: subprocess.Popen | None = None
        self._http = requests.Session()  # keep-alive: saves ~5-20 ms per command

        if not self._server_alive():
            if l.get("autostart", True):
                self._spawn_server()
            else:
                raise ConnectionError(
                    f"llama-server not reachable at {self.base} and autostart is disabled."
                )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _server_alive(self, timeout: float = 0.5) -> bool:
        """Return True if llama-server responds healthy at /health."""
        try:
            r = self._http.get(f"{self.base}/health", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _spawn_server(self, ready_timeout_s: float = 120.0) -> None:
        """Launch llama-server as a subprocess and block until it reports healthy."""
        server_bin = resolve(self._cfg["server_bin"])
        model = resolve(self._cfg["model_gguf"])
        if not server_bin.exists():
            raise FileNotFoundError(
                f"{server_bin} not found - run ./bootstrap.sh (llama.cpp build step)."
            )
        if not model.exists():
            raise FileNotFoundError(
                f"{model} not found - run ./bootstrap.sh (model download step)."
            )
        cmd = [
            str(server_bin),
            "--model",
            str(model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n-gpu-layers",
            str(self._cfg["n_gpu_layers"]),
            "--ctx-size",
            str(self._cfg["ctx_size"]),
            "--parallel",
            str(self._cfg.get("parallel", 1)),
        ]
        if self._cfg.get("no_warmup", False):
            cmd.append("--no-warmup")
        # Smaller compute ubatches keep each GPU submission short, so a 3-D
        # render pass sharing the iGPU waits less behind prompt processing.
        if self._cfg.get("ubatch_size"):
            cmd += ["--ubatch-size", str(int(self._cfg["ubatch_size"]))]
        cmd += list(self._cfg.get("extra_args", []) or [])

        # Deterministic GPU init on Strix Halo (gfx1151): the ROCm runtime can
        # segfault on the first kernel dispatch unless the iGPU is presented as
        # its nearest supported relative. Allow overriding/clearing via config.
        env = os.environ.copy()
        gfx = self._cfg.get("hsa_override_gfx_version", "11.5.1")
        if gfx:
            env.setdefault("HSA_OVERRIDE_GFX_VERSION", str(gfx))
        # Optional compute-unit fence (e.g. "0:0-31" leaves 8 of 40 CUs to the
        # graphics queue). Off by default: measure with so101_sim.bench first.
        if self._cfg.get("cu_mask"):
            env["HSA_CU_MASK"] = str(self._cfg["cu_mask"])

        logger.info("Starting llama-server (ROCm): %s", " ".join(cmd))
        log_dir = resolve(self._log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._server_log_path = log_dir / "llama-server.log"
        self._server_log = open(self._server_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdout=self._server_log, stderr=subprocess.STDOUT, env=env
        )
        deadline = time.time() + ready_timeout_s
        while time.time() < deadline:
            if self._server_alive():
                logger.info("llama-server ready at %s", self.base)
                return
            if self._proc.poll() is not None:
                tail = self._read_server_log_tail()
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode} during "
                    f"startup. Last log lines:\n{tail}\n"
                    f"(full log: {self._server_log_path})"
                )
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become healthy in time.")

    def _read_server_log_tail(self, n: int = 20) -> str:
        """Return the last n lines of the server log, for error messages."""
        try:
            with open(self._server_log_path) as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return "(no server log available)"

    def close(self) -> None:
        """Terminate the server if this object spawned it."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def parse(self, transcript: str) -> ParsedCommand:
        """Map one spoken transcript to a ParsedCommand (grammar-constrained)."""
        if not transcript or not transcript.strip():
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
        payload = {
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": transcript.strip()},
            ],
            "temperature": self.temperature,
            "max_tokens": 48,
            "grammar": self.grammar,
        }
        t0 = time.perf_counter()
        r = self._http.post(
            f"{self.base}/v1/chat/completions", json=payload, timeout=30
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        latency_ms = (time.perf_counter() - t0) * 1e3
        try:
            data = json.loads(content)
            command = data["command"]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Unparseable LLM output: %r", content)
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
        intent = Intent.from_string(command)
        params: dict[str, str] = {}
        obj = str(data.get("object", "") or "").strip()
        if obj:
            params["object"] = obj
        logger.info(
            "Intent %r %s ← %r (%.0f ms)",
            intent.value,
            params or "",
            transcript,
            latency_ms,
        )
        return ParsedCommand(intent, params=params, transcript=transcript)


# =============================================================================
# Standalone component test
# =============================================================================

_SELFTEST_CASES = [
    ("can you copy my arm movements", Intent.GESTURE_MIMIC, None),
    ("mirror my hand please", Intent.GESTURE_MIMIC, None),
    ("go grab the block and bring it to me", Intent.FETCH_BLOCK, None),
    ("fetch the cube", Intent.FETCH_BLOCK, None),
    ("pick up the ball", Intent.PICK_PLACE, "ball"),
    ("grab the bottle and hand it to me", Intent.PICK_PLACE, "bottle"),
    ("show me your best dance moves", Intent.DANCE, None),
    ("hold this pencil for me", Intent.GRIP, None),
    ("release", Intent.RELEASE, None),
    ("open your hand", Intent.RELEASE, None),
    ("drop it", Intent.RELEASE, None),
    ("wave hello", Intent.WAVE, None),
    ("go back to your home position", Intent.HOME, None),
    ("stop right now", Intent.STOP, None),
    ("what's the weather like today", Intent.UNKNOWN, None),
]


def main() -> None:
    """CLI entry point: parse one utterance, run the selftest suite, or start an interactive REPL."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Llama-3.2-3B intent parser (llama.cpp + ROCm) - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default=None, help="single utterance to parse")
    parser.add_argument(
        "--selftest", action="store_true", help="run the built-in utterance suite"
    )
    args = parser.parse_args()

    llm = LlamaIntentParser(load_config(args.config))
    try:
        if args.selftest:
            passed = 0
            for text, expected, obj in _SELFTEST_CASES:
                got = llm.parse(text)
                ok = got.intent == expected and (obj is None or got.object_name == obj)
                passed += ok
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {text!r} -> {got.intent.value} "
                    f"{got.params} (expected {expected.value} {obj or ''})"
                )
            print(f"\n{passed}/{len(_SELFTEST_CASES)} cases passed")
            raise SystemExit(0 if passed == len(_SELFTEST_CASES) else 1)
        if args.text:
            cmd = llm.parse(args.text)
            print(cmd.intent.value, cmd.params)
            return
        print("Interactive intent REPL - Ctrl-D to exit")
        while True:
            try:
                line = input("you> ")
            except EOFError:
                break
            cmd = llm.parse(line)
            print(f"  -> {cmd.intent.value} {cmd.params}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
