# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Measure where the twin's time goes - and what it costs llama-server.

    python -m so101_sim.bench                     # physics + rollout + render
    python -m so101_sim.bench --contention        # + iGPU sharing with llama-server
    python -m so101_sim.bench --contention --config config/pipeline.yaml

physics     real-time loop budget: mj_step + servo profile + current model
rollout     raw mj_step throughput, 1 process vs. N processes (CPU cores)
render      one tiled pass of the three standard views (wrist, operator, 3-D)
contention  llama-server intent-parse latency with/without the renderer
            running, and render pass time with/without LLM requests - the
            numbers behind llm.ubatch_size / llm.cu_mask in pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import threading
import time
import urllib.request


def bench_physics(seconds: float = 2.0) -> dict:
    from so101_sim.arm import SimArm
    from so101_sim.world import SimWorld

    w = SimWorld(start=False)
    arm = SimArm(w)
    arm.send_joints({"shoulder_pan": 60, "elbow_flex": 30, "gripper": 90})
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        w.advance(0.1)
        n += int(round(0.1 / w.dt))
    dt = time.perf_counter() - t0
    w.close()
    rate = n / dt
    return {"steps_per_s": round(rate), "realtime_x": round(rate * w.dt, 1),
            "core_pct_at_500hz": round(100.0 * 500.0 / rate, 2)}


def _rollout_worker(args) -> int:
    path, seconds = args
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.3
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(200):
            mujoco.mj_step(m, d)
        n += 200
    return n


def bench_rollout(seconds: float = 2.0, procs: int | None = None) -> dict:
    from so101_sim.config import DEFAULT_SCENE

    procs = procs or os.cpu_count() or 1
    single = _rollout_worker((str(DEFAULT_SCENE), seconds)) / seconds
    with mp.get_context("spawn").Pool(procs) as pool:
        counts = pool.map(_rollout_worker, [(str(DEFAULT_SCENE), seconds)] * procs)
    total = sum(counts) / seconds
    return {"single_steps_per_s": round(single), "procs": procs,
            "parallel_steps_per_s": round(total), "scaling_x": round(total / single, 1)}


def _render_rig(require_hw=False):
    from so101_sim.camera import SimCamera
    from so101_sim.render import RenderService
    from so101_sim.world import SimWorld

    w = SimWorld(start=True)
    rs = RenderService(w, require_hw=require_hw)
    cams = [SimCamera(w, r, renderer=rs, blocking=False) for r in ("arm", "operator", "view")]
    return w, rs, cams


def bench_render(seconds: float = 3.0) -> dict:
    w, rs, cams = _render_rig()
    try:
        time.sleep(0.5)
        samples = []
        p0 = rs.passes
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            samples.append(rs.pass_ms)
            time.sleep(1 / 60)
        passes = rs.passes - p0
        return {"renderer": rs.renderer, "views": "640x480 wrist + 640x480 operator + 800x800 3-D",
                "pass_ms_median": round(statistics.median(samples), 2),
                "pass_ms_p95": round(sorted(samples)[int(0.95 * (len(samples) - 1))], 2),
                "passes_per_s": round(passes / seconds, 1)}
    finally:
        rs.close()
        w.close()


def _llm_request(base: str, prompt: str) -> float:
    body = json.dumps({"prompt": prompt, "n_predict": 48, "temperature": 0.0, "cache_prompt": True}).encode()
    req = urllib.request.Request(f"{base}/completion", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    return 1000.0 * (time.perf_counter() - t0)


def bench_contention(cfg: dict | None, n: int = 12) -> dict:
    llm = (cfg or {}).get("llm") or {}
    base = f"http://{llm.get('host', '127.0.0.1')}:{llm.get('port', 8081)}"
    try:
        urllib.request.urlopen(f"{base}/health", timeout=2).read()
    except Exception as e:
        return {"error": f"llama-server not reachable at {base} ({e}); start it first "
                "(the pipeline autostarts it, or workshop/start_llama_server.sh)"}
    prompt = ("You map robot voice commands to JSON intents.\nCommand: pick up the red ball\nJSON:")
    _llm_request(base, prompt)  # warm the prompt cache
    alone = [_llm_request(base, prompt) for _ in range(n)]

    w, rs, cams = _render_rig()
    try:
        time.sleep(0.5)
        render_alone = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            render_alone.append(rs.pass_ms)
            time.sleep(1 / 60)
        with_render = [_llm_request(base, prompt) for _ in range(n)]
        render_busy = []
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                render_busy.append(rs.pass_ms)
                time.sleep(1 / 60)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        for _ in range(n):
            _llm_request(base, prompt)
        stop.set()
        th.join()
    finally:
        rs.close()
        w.close()
    med = statistics.median
    return {
        "llm_ms_alone": round(med(alone), 1),
        "llm_ms_with_render": round(med(with_render), 1),
        "llm_slowdown_pct": round(100.0 * (med(with_render) / med(alone) - 1.0), 1),
        "render_pass_ms_alone": round(med(render_alone), 2),
        "render_pass_ms_with_llm": round(med(render_busy), 2) if render_busy else None,
        "hint": "if render_pass_ms_with_llm > 33 ms, try llm.ubatch_size: 256, then llm.cu_mask: '0:0-31'",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 twin benchmarks")
    ap.add_argument("--contention", action="store_true", help="also measure iGPU sharing with llama-server")
    ap.add_argument("--config", default=None, help="pipeline.yaml (llm host/port for --contention)")
    ap.add_argument("--only", choices=["physics", "rollout", "render"], default=None)
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    cfg = None
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    results = {}
    if args.only in (None, "physics"):
        results["physics"] = bench_physics()
    if args.only in (None, "rollout"):
        results["rollout"] = bench_rollout(procs=args.procs)
    if args.only in (None, "render"):
        results["render"] = bench_render()
    if args.contention:
        results["contention"] = bench_contention(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
