# Ryzen AI VVLA workshop - hands-on materials

A 3-hour workshop that takes a CS/engineering student from zero to a
running **vision -> language -> action** pipeline on an AMD Ryzen AI
APU driving a LeRobot SO-101 arm: YOLOv26s x2 on the **NPU** (VitisAI EP),
Whisper on the **CPU**, Llama 3.2 3B on the **iGPU** (llama.cpp + ROCm),
MediaPipe Hands on the **CPU**, all wired over **ROS 2**.

The workshop depends on [Installation](../README.md#installation). That
creates `.venv`, the model files in `../models`, and the YOLO NPU compile
cache in `../cache`. The two YOLO models load from that cache. Whisper
runs on the CPU.

## What's in the box

```
notebooks/               phase 1 - read & run these first (each is self-contained)
  01_ai_models/          NPU session factory - MediaPipe on CPU - Llama on iGPU
  02_ros2/               camera nodes/clients - SO-101 server/client - Llama intent node
  03_integration/        registry dispatch - the mimic composition - the live UI -
                         the digital twin (MuJoCo arm, cameras, grasp; live console over ROS 2)
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

Before the steps below, install the Ryzen AI SDK and prerequisites as described in [Installation](../README.md#installation).

```bash
cd workshop
./run_notebooks.sh                        # registers the kernel, installs missing
                                          # deps, starts the llama server and the
                                          # resource HUD, then serves Jupyter Lab.
                                          # This shell stays here until you quit it.
```

Jupyter opens `notebooks/`. Read and run them in order: `01_ai_models`,
`02_ros2`, `03_integration`.

The `02_ros2` notebooks and section 5 of `03_integration/02_digital_twin.ipynb` import `rclpy`. `bootstrap.sh` installs ROS 2 Jazzy and creates `.venv` with `--system-site-packages`. `./run_notebooks.sh` sources `/opt/ros/jazzy/setup.bash` before Jupyter starts.

After the notebooks, the TODOs are in `project/`. Follow [project/README.md](project/README.md).

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

`./run_notebooks.sh --test` runs every notebook and the commands above
headless, after the same setup as the happy path. See
[Automated tests](../README.md#automated-tests).

AMD-specific lessons for the session:

**Choose the device on purpose.** YOLO-pose and YOLO-detect already fill
the NPU. Another model, such as Whisper, can run on the CPU while those
two stay there. Pass `device="cpu"` or `device="npu"`, then print the
provider the session got in order to verify where it landed.

Generated dirs (`_artifacts/`, `logs/`, `__pycache__/`) are disposable -
see `.gitignore`.
