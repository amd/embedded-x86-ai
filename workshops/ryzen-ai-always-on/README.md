<!--
Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Portions of this file consist of AI-generated content. AI-assisted
content has been reviewed and validated by the authors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-->

# Always-On Physical AI with AMD Ryzen™ AI Software

This repo hosts the Jupyter Notebook, supporting modules, and assets for the "Always-On Physical AI with AMD Ryzen™ AI Software" workshop for AMD Advancing AI 2026.

## License

This project is licensed under the BSD 3-Clause License — see the [`LICENSE`](LICENSE) file for details.

## Repo structure

- `notebook/always_on_physical_ai.ipynb` — workshop Jupyter Notebook
- `notebook/workshop_utils/` — supporting modules
  - `yolo_pipeline.py` — YOLO26 pre/postprocessing
  - `ep_factory.py` — helper functions that create ONNX Runtime sessions for the CPU and NPU
  - `resource_monitor.py` — reads CPU/RAM/iGPU/NPU usage, power, and temperature
  - `live_dashboard.py` — live dashboard of resource monitor stats in the Notebook
  - `partition_viz.py` — NPU partition visualization
  - `pose_common.py` — shared helpers for the live webcam & teleop demos (decode NPU pose output, draw the skeleton, open the camera)
  - `threaded_inference.py` — runs the live webcam/video demo (NPU or CPU) on background threads
  - `hand_landmarks.py` — crops, runs, and decodes the 21-point hand model on the NPU
  - `hand_to_arm.py` — maps hand landmarks to SO-101 tele-op command
  - `arm_ik.py` — inverse kinematics code
  - `arm_teleop_sim.py` — SO-101 MuJoCo simulation renderer (iGPU)
  - `arm_teleop_mirror.py` — pinch-to-teleop pipeline
  - `pipeline_registry.py` — keeps a single live pipeline running at a time
- `compile.py` — compile a model for the NPU from a terminal (watch the VAIML compile and pre-warm the cache)
- `models/hand_landmark.onnx` — pre-built 21-point BlazeHand model (bundled so Section 8 works offline, no runtime download)
- `demo_yolo26s.py` — standalone live pose demo (webcam or video, NPU or CPU)
- `demo_teleop.py` — standalone pinch-to-teleop demo (pose + hand on NPU, SO-101 in MuJoCo, live dashboard)
- `robots/so101/` — SO-101 model (from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)) for the tele-op demo
- `assets/` — sample image (`sample.jpg`) and videos (`sample_video.mp4`, `teleop_sample.mp4`)
- `provisioning/` — one-time setup (systemd service that sets NPU turbo at boot)
  - `install-npu-turbo.sh` — installs and enables the turbo-at-boot service
  - `npu-turbo.service` — systemd unit that runs `xrt-smi configure --pmode turbo`
- `requirements.txt` — pip extras on top of the Ryzen AI venv (Linux)
- `validate_env.py` — environment setup test



## Tested environment

| Component | Version |
|---|---|
| Hardware | AMD Ryzen AI Max+ PRO 395 (Strix Halo), XDNA 2 NPU |
| OS | Ubuntu 24.04 |
| Ryzen AI SW | 1.7.1 |

## Install Ryzen AI Software 1.7.1

Reference: [Linux Installation Instructions — Ryzen AI Software 1.7.1 documentation](https://ryzenai.docs.amd.com/en/1.7.1/linux.html). The same driver steps are in [workshops/vvla-pipeline/README.md](../vvla-pipeline/README.md#installation).

### 1. Compatible hardware
You need an AMD Ryzen AI processor with an XDNA NPU (Ryzen AI 300 series or later).
Confirm the NPU is present with:
```bash
lspci -nn | grep 1022:17f0    # STX/KRK NPU signature
```

### 2. System prerequisites
**Requirements:** Ubuntu 24.04 LTS, kernel ≥ 6.10, Python 3.12.x, 64GB RAM recommended.
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv libboost-filesystem1.74.0 dkms
```

### 3. Downloads

- `RAI_1.7.1_Linux_NPU_XRT.zip` — [NPU drivers](https://account.amd.com/en/forms/downloads/xef.html?filename=RAI_1.7.1_Linux_NPU_XRT.zip)
- `ryzen_ai-1.7.1.tgz` — [Ryzen AI Software](https://account.amd.com/en/forms/downloads/xef.html?filename=ryzen_ai-1.7.1.tgz)

### 4. Install the NPU drivers (XRT)

The zip may unpack into a subdirectory, so the `find` lands in the directory that holds the `.deb` files.

```bash
cd ~/Downloads
unzip -o RAI_1.7.1_Linux_NPU_XRT.zip -d RAI_1.7.1_Linux_NPU_XRT
cd "$(dirname "$(find RAI_1.7.1_Linux_NPU_XRT -name 'xrt_202610.2.21.75_24.04-amd64-base.deb' -print -quit)")"
sudo apt install --fix-broken -y ./xrt_202610.2.21.75_24.04-amd64-base.deb
sudo apt install --fix-broken -y ./xrt_202610.2.21.75_24.04-amd64-base-dev.deb
sudo apt install --fix-broken -y ./xrt_202610.2.21.75_24.04-amd64-npu.deb
sudo apt install --fix-broken -y ./xrt_plugin.2.21.260102.53.release_24.04-amd64-amdxdna.deb
```

### 5. Set the environment and verify the driver sees the NPU

`xrt-smi examine` must list a device under `Device(s) Present`. The name depends on the machine (`RyzenAI-npu5`, `NPU Strix`, `NPU Strix Halo`). If the table is empty, stop. Do not install the software or compile a model.

```bash
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

### 6. Install Ryzen AI Software 1.7.1 (creates the venv)
```bash
mkdir -p ryzen_ai-1.7.1 && cp ryzen_ai-1.7.1.tgz ryzen_ai-1.7.1/
cd ryzen_ai-1.7.1 && tar -xvzf ryzen_ai-1.7.1.tgz
./install_ryzen_ai.sh -a yes -p $HOME/ryzenai/venv
source $HOME/ryzenai/venv/bin/activate
echo $RYZEN_AI_INSTALLATION_PATH      # sanity check
```

Verify the VitisAI EP is available:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Expected: `['VitisAIExecutionProvider', 'CPUExecutionProvider']`


### 7. Test the NPU install
```bash
cd $HOME/ryzenai/venv/quicktest && python quicktest.py    # output "Test Finished"
```

### 8. Set turbo automatically at boot (one time)
The NPU runs at its full clock only in **turbo** power mode, and it resets to
`Default` on every reboot. Rather than run
`sudo xrt-smi configure --pmode turbo` before every session, install a small
systemd service once. It runs the command as root on every boot, so the
workshop itself needs no `sudo`:

```bash
sudo ./provisioning/install-npu-turbo.sh
```

This copies [`provisioning/npu-turbo.service`](provisioning/npu-turbo.service)
to `/etc/systemd/system/` and enables it, so the machine boots straight into
turbo with no manual step.

Verify it took effect:
```bash
systemctl status npu-turbo.service           # should be "active (exited)"
xrt-smi examine -r platform | grep -i mode    # should report turbo
```

> **Note:** the final teleop demo (Section 9) drives the NPU, iGPU, and CPU at
> once. With the NPU in turbo (max clock), some hardware exceeds its thermal
> envelope and resets. If that happens, drop the NPU to default power mode first:
> ```bash
> sudo xrt-smi configure --pmode default
> ```

## Reactivate environment 
Every new shell needs both the XRT environment (for the NPU) and the venv (for Python):

```bash
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
source /opt/xilinx/xrt/setup.sh
source $HOME/ryzenai/venv/bin/activate
```

## Notebook quick start
### 1. Clone the workshop repo
```bash
git clone https://github.com/amd/embedded-x86-ai.git
cd embedded-x86-ai/workshops/ryzen-ai-always-on
```

### 2. Install workshop requirements
```bash
source $HOME/ryzenai/venv/bin/activate    # make sure venv is activated
pip install -r requirements.txt
pip install ultralytics                   # required for the YOLO26 export
pip install "numpy==1.26.4"               # ultralytics bumps numpy to 2.x; the NPU stack (onnxruntime/flexml/quark) needs 1.26.4
```

### 3. Validate setup
```bash
python validate_env.py    # All checks should print `OK`.
```

### 4. Launch Jupyter Notebook
```bash
jupyter lab notebook/always_on_physical_ai.ipynb
```

## Standalone demos

These self-contained scripts run outside Jupyter in an OpenCV window. Both reuse
the VAIML compile caches under `cache/` (built once by the notebook), so on the
NPU they start in ~1s with no recompile. Activate the venv (and, for the NPU,
source the XRT environment — see [Reactivate environment](#reactivate-environment))
first.

### `demo_yolo26s.py` — live pose

Runs yolo26s-pose on a webcam or video file and draws a live skeleton overlay
with an FPS counter. Supports NPU or CPU.

```bash
# NPU (BF16) — webcam
python demo_yolo26s.py

# CPU (FP32) — webcam
python demo_yolo26s.py --cpu

# Video file
python demo_yolo26s.py --video assets/sample_video.mp4

# Set the webcam capture resolution (default 1920x1080)
python demo_yolo26s.py --width 1280 --height 720
```

### `demo_teleop.py` — pinch-to-teleop (the "all three units lit" finale)

The same demo as the notebook's Section 8, standalone. Your hand flies a
simulated SO-101 arm; pinch to grab and place a cube. All three compute units run
at once, with the live resource dashboard stacked under the feed:

- **NPU** — yolo26s-pose finds the presenter and crops a hand box from the wrist; the 21-point hand-landmark model then reads the fingers in that crop. Both run on the AIE
- **CPU** — hand landmarks → end-effector target + pinch (vector math)
- **iGPU** — MuJoCo SO-101 render (EGL) with damped-least-squares IK

```bash
# NPU — webcam + dashboard (default)
python demo_teleop.py

# Run both models on the CPU (no NPU/turbo needed; much lower fps)
python demo_teleop.py --cpu

# Video file, or hide the dashboard
python demo_teleop.py --video assets/sample_video.mp4
python demo_teleop.py --no-dashboard
```

Controls: `q` quit, `r` reset cube.

Both demos: press `q` to quit.
