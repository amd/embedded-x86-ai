# Ryzen AI VVLA workshop - hands-on materials

A 3-hour workshop that takes a CS/engineering student from zero to a
running **vision -> language -> action** pipeline on an AMD Ryzen AI
APU driving a LeRobot SO-101 arm: YOLOv26s x2 on the **NPU** (VitisAI EP),
Llama 3.2 3B on the **iGPU** (llama.cpp + ROCm), MediaPipe Hands on the
**CPU**, all wired over **ROS 2**.

Everything here is **self-contained**: no imports from the reference
pipeline package anywhere. The workshop shares only the repo's model files
and NPU compile cache (`../models`, `../cache`), so nothing recompiles.

## What's in the box

```
notebooks/               phase 1 - read & run these first (each is self-contained)
  01_ai_models/          NPU session factory - MediaPipe on CPU - Llama on iGPU
  02_ros2/               camera nodes/clients - SO-101 server/client - Llama intent node
  03_integration/        registry dispatch - the mimic composition - the live UI
common/                  PROVIDED helpers (imaging, motion, ROS codecs, fixtures,
                         reporting, resource HUD) - one shared copy, imported below
project/                 phase 2 - prebuilt project with 21 TODO stubs you implement
  models/  ros/  integration/  app.py   your code + prebuilt harnesses
solution/                identical tree, every TODO filled in (the answer key)
launch_monitor.sh        always-on-top resource HUD: CPU% · GPU% · NPU inf/s | Idle
MONITOR.md               how to open/close that HUD (from a shell or a notebook)
```

**The rule that decided what's a TODO:** if code would be byte-for-byte
identical in any non-AMD, non-ROS project (letterboxing, decoders,
overlays, message codecs, IK/smoothing, grammar files), it's provided. If
it's where code meets the **Ryzen AI NPU/iGPU**, the **ROS transport**, or
the **pipeline's control flow**, it's yours.

## Student happy path

Before step 0, install the Ryzen AI SDK and prerequisites as described in [Installation](../README.md#installation).

```bash
./bootstrap.sh                            # 0) once, from the repo root: builds the
                                          #    .venv the workshop kernel runs on,
                                          #    exports the YOLO models, and compiles
                                          #    them for the NPU (~20 min for both YOLOs)
cd workshop
./run_notebooks.sh                        # 1) registers the kernel on that venv,
                                          #    installs missing deps, starts the
                                          #    llama server, opens the resource HUD and Jupyter

cd project
python -m models.selftest                 # 2) component 1: iterate until PASS
python -m ros.selftest                    # 3) component 2: offline round-trips
python app.py --dry-run --synthetic --no-llm    # 4) component 3: the live UI
python app.py                             # 5) the rig: NPU + ROS 2 + Llama + arm
```

The ROS 2 notebooks (`02_ros2/`) run on real ROS 2 only. If `rclpy` isn't
importable they stop and print install instructions; install ROS 2 Jazzy
and relaunch `./run_notebooks.sh` so the kernel picks it up.

Each harness prints per-TODO status (PASS / FAIL / TODO / SKIP) with
targeted hints and drops proof into `_artifacts/` (pose overlay, camera
round-trip frame, UI snapshot). Off-hardware, the models and app degrade
gracefully: CPU fallback, synthetic camera, scripted person/hand, dry-run
arm.

## Live resource monitor

A tiny always-on-top window shows the three numbers that matter while a cell
runs - **CPU %** (burgundy), **GPU %** (orange), and **NPU inferences/sec or
Idle** (cyan) - floating over the browser and everything else. `run_notebooks.sh`
opens it for you (pass `--no-monitor` to skip); to open or reopen it on its own:

```bash
./launch_monitor.sh                          # from the workshop folder
```

```python
from common.monitor import open_monitor, close_monitor
open_monitor()       # floats on top; close_monitor() dismisses it
```

Drag to move it; close it with its ✕, the Esc key, or a right-click. It reads
`top`, the amdgpu `gpu_busy_percent` sysfs gauge (`radeontop` fallback), and
`xrt-smi`; any tool that's missing just shows
`N/A` / `Idle`, so it runs on a plain laptop too. Details in `MONITOR.md`.

## Instructor's corner

```bash
diff -r project solution                  # exactly the 21 TODO bodies, nothing else
cd solution && python -m models.selftest && python -m ros.selftest \
  && python app.py --dry-run --synthetic --no-llm --headless --frames 60
```

The two AMD-specific lessons to hammer home during the session:

1. **Verify the execution provider.** Requesting the NPU isn't getting it.
   A silent CPU fallback runs correctly and 10-30x slower. Session built ->
   provider printed. Always.
2. **The NPU context budget.** Too many resident models for the concurrent
   XDNA2 hardware contexts fails with `HW context creation unsuccessful`.
   It's not a stale cache; move a model to CPU deliberately (camera-rate
   vision stays on the NPU).

Generated dirs (`_artifacts/`, `logs/`, `__pycache__/`) are disposable -
see `.gitignore`.
