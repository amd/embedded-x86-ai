# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Gemma 4 E4B intent parsing via llama.cpp running on ROCm.

The model runs in ``llama-server`` (built with ``-DGGML_HIP=ON`` by
``bootstrap.sh``) and is queried over its OpenAI-compatible HTTP API. A GBNF
grammar (``config/intent.gbnf``) constrains generation so the model can only
emit the list ``["<name>", "<object>", "<words heard>"]`` - no free-form
parsing.

A request returns a :class:`ParsedCommand` (intent **plus parameters**) so
commands like "pick up the ball" carry their object through to behaviors.

The answer is read as the server writes it (a stream of server-sent events).
It is written in the order command, object, heard, so the call is complete
before the words heard are: ``on_call`` receives the call at that moment, with
the seconds since the request began, while the rest is still being written.
Nothing waits for the whole answer to act on the call. A typed sentence is its
own words, so its request ends at the call.

Independent test
----------------
    python -m vla_pipeline.llm.llama_intent --text "hey can you pick up the ball"
    python -m vla_pipeline.llm.llama_intent --selftest
    python -m vla_pipeline.llm.llama_intent           # interactive REPL
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator

import requests

from vla_pipeline.llm.intents import (
    INTENT_SYSTEM_PROMPT,
    Intent,
    ParsedCommand,
    call_from_partial,
    heard_from_partial,
)
from vla_pipeline.utils.config import load_config, resolve

logger = logging.getLogger(__name__)

# What llama-server logs once its model is loaded and it serves requests.
READY_LINE = "listening on http://"
# The longest a request may go without the server sending anything, in seconds.
REQUEST_TIMEOUT_S = 30
# The call, and the seconds from the request to it. Returning False says the words are not wanted.
OnCall = Callable[[ParsedCommand, float], bool | None]


def answer_pieces(lines) -> Iterator[str]:
    """The answer's text piece by piece, as the server's events carry it.

    ``lines`` are the lines of an OpenAI-style event stream; it ends at ``[DONE]``.
    """
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"llama-server reported: {str(event['error'])[:300]}")
        choices = event.get("choices") or [{}]
        yield (choices[0].get("delta") or {}).get("content") or ""


class LlamaIntentParser:
    """Manage a llama.cpp server and turn transcripts into :class:`ParsedCommand`."""

    def __init__(self, cfg: dict, *, section: str = "gemma4-e4b-it"):
        l = cfg[section]
        self._audio = bool(l.get("mmproj"))
        self.host: str = l["host"]
        self.port: int = int(l["port"])
        self.base = f"http://{self.host}:{self.port}"
        self.temperature: float = float(l.get("temperature", 0.0))
        self.grammar = resolve(l["grammar_file"]).read_text(encoding="utf-8")
        self._cfg = l
        self._log_dir = cfg.get("system", {}).get("log_dir", "logs")
        self._proc: subprocess.Popen | None = None
        self._http = requests.Session()  # keep-alive: saves ~5-20 ms per command
        self.last_call_s: float | None = None  # the last request: seconds until its call was complete
        self.last_total_s: float | None = None  # the last request: seconds until reading its answer ended

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
        """Launch llama-server as a subprocess and block until it logs that it
        is listening (:data:`READY_LINE`), or exits."""
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
        if self._cfg.get("mmproj"):
            mmproj = resolve(self._cfg["mmproj"])
            if not mmproj.exists():
                raise FileNotFoundError(
                    f"{mmproj} not found - the Gemma audio projector has not finished downloading."
                )
            cmd += ["--mmproj", str(mmproj)]
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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, errors="replace",
        )
        ready = threading.Event()

        def relay() -> None:
            """The server's output to its log; its readiness line, or its exit, sets ``ready``."""
            for line in self._proc.stdout:
                self._server_log.write(line)
                self._server_log.flush()
                if READY_LINE in line:
                    ready.set()
            ready.set()

        threading.Thread(target=relay, daemon=True, name="llama-server-log").start()
        try:
            if not ready.wait(ready_timeout_s):
                raise TimeoutError(
                    f"llama-server did not log {READY_LINE!r} within {ready_timeout_s:.0f} s "
                    f"(log: {self._server_log_path})"
                )
            if self._proc.poll() is not None:
                tail = self._read_server_log_tail()
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.wait()} during "
                    f"startup. Last log lines:\n{tail}\n"
                    f"(full log: {self._server_log_path})"
                )
            if not self._server_alive():
                raise RuntimeError(f"llama-server logged readiness but {self.base}/health is not ready")
        except BaseException:
            self.close()
            raise
        logger.info("llama-server ready at %s", self.base)

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
        server_log = getattr(self, "_server_log", None)
        if server_log is not None and not server_log.closed:
            server_log.close()

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def _complete(self, user_content, transcript: str, on_call: OnCall | None, words: bool) -> ParsedCommand:
        """One chat completion, grammar-constrained, its answer read as it is written.

        ``on_call`` gets the call the moment it is complete. With ``words`` false, or when
        ``on_call`` returns False, the reading ends there and the connection is closed: for a
        caller that has the sentence already, or that drops the call. The server is then free
        for the next request sooner. An answer that ends before its call is complete is
        ``unknown``.
        """
        payload = {
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": 96,  # the command list plus the short "heard" phrase
            "grammar": self.grammar,
            "stream": True,
        }
        if self._audio:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        self.last_call_s = self.last_total_s = None
        # The seconds of this request are kept in locals: a typed sentence and a spoken one can
        # be asked from two threads, and the fields above are only the last request's record.
        text, call, call_s = "", None, 0.0
        t0 = time.perf_counter()
        with self._http.post(
            f"{self.base}/v1/chat/completions", json=payload, stream=True, timeout=REQUEST_TIMEOUT_S
        ) as r:
            if r.status_code >= 400:
                logger.error("llama-server rejected the request: %s", r.text[:500])
            r.raise_for_status()
            # chunk_size=1: a line is handed on when its last byte arrives. The library's
            # default reads in blocks of 512 bytes and would hold the end of the call back
            # until the server has written that much more.
            for piece in answer_pieces(r.iter_lines(chunk_size=1)):
                text += piece
                if call is None:
                    call = call_from_partial(text)
                    if call is not None:
                        call_s = self.last_call_s = time.perf_counter() - t0
                        # on_call answers False when it has no use for the words (a call it drops).
                        wanted = on_call(call, call_s) if on_call is not None else None
                        if not words or wanted is False:
                            break  # leaving the block closes the connection; the rest is not read
        total_s = self.last_total_s = time.perf_counter() - t0
        if call is None:
            logger.warning("Unparseable LLM output: %r", text)
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
        heard = heard_from_partial(text) or transcript
        logger.info(
            "Intent %r %s ← %r (call after %.0f ms, %.0f ms in all)",
            call.intent.value,
            call.params or "",
            heard,
            call_s * 1e3,
            total_s * 1e3,
        )
        return ParsedCommand(call.intent, params=call.params, transcript=heard)

    def parse(self, transcript: str, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one typed sentence to a ParsedCommand (grammar-constrained).

        The sentence is its own words, so the request ends when the call is complete.
        """
        if not transcript or not transcript.strip():
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
        return self._complete(transcript.strip(), transcript, on_call, words=False)

    def parse_audio(self, wav: bytes, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one WAV clip to a ParsedCommand. The clip is the user message.

        ``on_call`` receives the call while Gemma still writes the words it heard; the
        returned command holds those words.
        """
        encoded = base64.b64encode(wav).decode("ascii")
        content = [
            {"type": "text", "text": "Classify the spoken robot instruction in this audio."},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
        ]
        return self._complete(content, "", on_call, words=True)

    def warm_up(self) -> float:
        """One request whose answer is thrown away; the seconds it took.

        The first request to a freshly started server pays for the instructions being read
        into the model. Paid here, at the start, not at the user's first command.
        """
        self.parse("hello")
        return float(self.last_total_s)


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
    ("go home", Intent.HOME, None),
    ("stop robot", Intent.STOP, None),
    ("what's the weather like today", Intent.UNKNOWN, None),
]


def main() -> None:
    """CLI entry point: parse one utterance, run the selftest suite, or start an interactive REPL."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Gemma 4 intent parser (llama.cpp + ROCm) - component test"
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
