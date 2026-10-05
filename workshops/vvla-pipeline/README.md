# Ryzen AI VVLA Workshop

Vision → Language → Action on an **AMD Ryzen AI APU** driving a **LeRobot SO-101** arm.
YOLOv26s pose and detection run on the **NPU** (VitisAI EP), Llama 3.2 3B runs on the
**iGPU** (llama.cpp + ROCm), MediaPipe Hands runs on the **CPU**, and everything is wired
together over **ROS 2**.

![Modular VLA pipeline](workshop/animations/modular_vla_pipeline.gif)

This README covers machine setup. Once setup is done, the hands-on material lives in
[`workshop/`](workshop/README.md).

## Tested environment

| Component | Version |
|---|---|
| Hardware | AMD Ryzen AI APU (Strix Point / Strix Halo), XDNA 2 NPU |
| OS | Ubuntu 24.04.4 |
| ROCm | 7.2.4 (installed by `bootstrap.sh` if absent) |
| Ryzen AI SW | 1.7.1 (VitisAI ONNX Runtime EP) |
| ROS 2 | Jazzy (installed by `bootstrap.sh` if absent) |
| LeRobot | 0.5.2 (source install, `[feetech]`) |
| Robot | LeRobot SO-101 follower arm |

## Compute placement

| Model | Role | Device | Runtime |
|---|---|---|---|
| YOLOv26s-pose | where the person is | **NPU** | ONNX Runtime, VitisAI EP |
| YOLOv26s-detect | what to grab | **NPU** | ONNX Runtime, VitisAI EP |
| Llama 3.2 3B Instruct (Q4_K_M) | intent | **iGPU** | `llama.cpp` `llama-server` (HIP) |
| MediaPipe Hands | pinch / gesture | **CPU** | MediaPipe |

## Installation

All commands below run from this directory (`workshops/vvla-pipeline/`) of your
`embedded-x86-ai` checkout.

### 1. Ryzen AI software and bootstrap

Download Ryzen AI 1.7.1 from
https://account.amd.com/en/forms/downloads/xef.html?filename=ryzen_ai-1.7.1.tgz
and untar it into this directory, so you have `workshops/vvla-pipeline/ryzen_ai-1.7.1/`.

```bash
cd embedded-x86-ai/workshops/vvla-pipeline
tar -xzf ~/Downloads/ryzen_ai-1.7.1.tgz

# Point at your Ryzen AI 1.7.1 wheel directory so the NPU EP gets installed.
# Without it, everything falls back to the CPU execution provider.
export RYZEN_AI_WHEELS=./ryzen_ai-1.7.1

./bootstrap.sh
```

`bootstrap.sh` does, in one pass:

1. apt build deps, plus ROCm and ROS 2 Jazzy if they are missing
2. a uv venv at `./.venv` with `--system-site-packages` (so ROS 2's `rclpy` stays importable)
3. Python deps from `requirements.txt`, PyTorch ROCm wheels, and LeRobot `[feetech]` → `third_party/lerobot`
4. llama.cpp HIP build (`llama-server`) → `third_party/llama.cpp`
5. the Ryzen AI onnxruntime (VitisAI EP) from `$RYZEN_AI_WHEELS`
6. the Llama-3.2-3B-Instruct Q4_K_M GGUF → `models/llama-3.2-3b/`

Flags: `--skip-apt`, `--skip-llama`, `--skip-models`, `--skip-compile`, `--cpu-only`
(dev machine without ROCm/NPU).

> **Note:** `meta-llama/Llama-3.2-3B-Instruct` is gated. If the GGUF download asks for
> authentication, run `hf auth login` and rerun.

### 2. Export the YOLO models

```bash
source .venv/bin/activate
uv pip install ultralytics 'numpy<2'
python scripts/export_yolo26s_pose.py      # → models/yolo26s-pose/yolo26s-pose.onnx
python scripts/export_yolo26s_detect.py    # → models/yolo26s/yolo26s.onnx
deactivate
```

### 3. Compile for the NPU

The NPU models must be compiled once. Compilation needs the full Ryzen AI SDK
compiler; the `.venv` runtime can only run models that are already compiled.
Rerunning bootstrap installs the compiler into `./ryzenai-compile/`, compiles both
YOLO models into `cache/`, and re-pins anything `ultralytics` changed in `.venv`:

```bash
sudo apt install -y linux-libc-dev zip
[ -d /usr/include/asm ] || sudo ln -s /usr/include/asm-generic /usr/include/asm
./bootstrap.sh --skip-apt --skip-llama --skip-models
```

This needs about 50 GB free for the SDK. To recompile later without bootstrap:

```bash
source ./ryzenai-compile/bin/activate
python scripts/compile_npu_models.py           # both; or --only yolo_pose / yolo_detect
deactivate
```

The compile reads model paths and cache keys from
`workshop/solution/config/workshop.yaml`, so the cache matches what the workshop loads.

### 4. Robot setup (hardware runs only)

```bash
lerobot-find-port                 # find the Feetech bus, e.g. /dev/ttyACM0
sudo chmod 666 /dev/ttyACM0       # resets on replug; see udev_rule_for_robot.sh for a permanent rule
```

Calibrate the arm once with [`arm_calibrator/`](arm_calibrator/README.md) (home pose
reference: [`calibration/`](calibration/)). Then set `robot.motor_port`, `robot.robot_id`
and the `cameras.*.device` paths in `workshop/project/config/workshop.yaml` and
`workshop/solution/config/workshop.yaml`.

## Run the workshop

```bash
cd workshop
./run_notebooks.sh
```

This registers the Jupyter kernel on `.venv`, starts `llama-server`, opens the resource
monitor, and launches Jupyter. Continue with [`workshop/README.md`](workshop/README.md).

## Directory layout

```
workshops/vvla-pipeline/
├── bootstrap.sh              # one-shot machine setup into ./.venv
├── requirements.txt
├── scripts/
│   ├── export_yolo26s_pose.py    # YOLO pose → ONNX
│   ├── export_yolo26s_detect.py  # YOLO detect → ONNX
│   ├── compile_npu_models.py     # compile the YOLO ONNX for the NPU (SDK venv)
│   └── ryzen_ai_env.sh           # LD_LIBRARY_PATH for the VitisAI EP runtime
├── arm_calibrator/           # SO-101 calibration
├── calibration/              # home-pose reference image + notes
├── udev_rule_for_robot.sh    # serial port permissions
└── workshop/                 # notebooks, project (TODOs) and solution - start here
```

Created during setup: `.venv/`, `third_party/`, `models/`, `cache/`, `ryzen_ai-1.7.1/`,
`ryzenai-compile/`.

## Troubleshooting

- **"VitisAI EP not available — falling back to CPU"**: set `RYZEN_AI_WHEELS` and rerun
  `./bootstrap.sh`, and confirm the XDNA driver is loaded (`ls /dev/accel/`).
- **"Model compilation is not supported in a deployment only installation"**: the YOLO
  models were never compiled. Run step 3.
- **`HW context creation unsuccessful` on the NPU**: too many models are resident on the
  NPU at once. Move one to the CPU by setting its `device: cpu` in `workshop.yaml`.
- **torch sees no GPU**: add yourself to the `render`/`video` groups
  (`sudo usermod -a -G render,video $USER`) and reboot.
- **llama-server fails to start**: check `third_party/llama.cpp/build/bin/llama-server`
  exists; rebuild with `./bootstrap.sh --skip-apt --skip-models`.
- **llama-server segfaults with `cudaMalloc failed: out of memory`**: keep `llm.ctx_size`
  capped (2048) and `llm.parallel: 1`; if still tight, lower `n_gpu_layers` or raise the
  UMA/VRAM split in BIOS.
- **llama-server segfaults during "warming up the model"**: a gfx1151 ROCm issue; keep
  `llm.no_warmup: true`.
- **Two identical USB cameras swap on reboot**: use the stable
  `/dev/v4l/by-id/...-video-index0` path for `cameras.*.device`.
- **Arm camera image is sideways**: set `cameras.arm.rotate` (0/90/180/270).
- **Arm doesn't move**: check `motor_port` permissions and that the arm is calibrated.
