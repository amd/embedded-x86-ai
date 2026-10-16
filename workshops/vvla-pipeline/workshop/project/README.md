# Workshop project - voice→vision→language→action on Ryzen AI

This is the hands-on project for the workshop. Everything generic is
prebuilt; the parts where code meets the **AMD Ryzen AI stack**, the
**ROS 2 transport**, and the **pipeline's control flow** are TODOs that you
implement. (In `solution/` every TODO is filled in - diff against it when
you're stuck or done.)

## Happy path

1. **Read and run the notebooks first** (`../notebooks/`, in order:
   `01_ai_models` → `02_ros2` → `03_integration`). They teach exactly what
   each TODO needs.
2. **Implement component 1** (`models/npu.py`, `models/hands.py`,
   `models/llm.py`), checking yourself as you go:

       python -m models.selftest

3. **Implement component 2** (`ros/camera_node.py`, `ros/camera_client.py`,
   `ros/arm_node.py`, `ros/arm_client.py`, `ros/intent_node.py`):

       python -m ros.selftest            # offline - works without ROS 2/hardware
       python -m ros.selftest --fake     # force the in-memory transport

4. **Implement component 3** (`integration/dispatch.py`,
   `integration/mimic.py`, `integration/behaviors.py`), checking yourself
   as you go, then launch the UI:

       python -m integration.selftest    # offline - scripted pose/hands, dry-run arm

       python app.py --dry-run --synthetic --no-llm    # anywhere
       python app.py                                   # the rig

Run everything **from this directory** (the packages resolve from here).

## Check everything at once

    python selftest.py

runs all three component harnesses against synthetic backends, tallies every
PASS / FAIL / TODO / SKIP into one scoreboard, and then re-runs them against
`solution/` to prove the harness itself is trustworthy. Read the result as:
**zero FAILs means nothing you've written so far conflicts with the solution -
the only gap left is the TODOs it lists.** A FAIL is different from a TODO: it
is code that ran and did the wrong thing, so fix it before moving on. Useful
flags: `--strict` exits nonzero until every TODO is done (good for "am I
finished?" and CI), `--quiet` prints just the scoreboards, `--real-ros` uses
the real ROS 2 transport instead of the in-memory shim.

`python app.py` is **fail-soft**: on a half-finished project it prints the
TODOs *this* run still needs - and how to shrink that set with
`--dry-run` / `--synthetic` / `--no-llm` - instead of dying on the first stub.
So the UI comes up the moment the pieces you actually need are in, and typing a
command whose behavior isn't written yet just says so instead of crashing.

Want to watch the hardware while the UI or a notebook runs? Pop the
always-on-top resource HUD - **CPU %**, **GPU %**, and **NPU inferences/sec**
(or Idle):

    ../launch_monitor.sh                       # from a shell, or in a notebook:
    from common.monitor import open_monitor    # open_monitor() / close_monitor()

## Layout

    config/        workshop.yaml (real rig values), intent.gbnf, vaiep_config.json
    common/        PROVIDED helpers - a pointer to the single shared copy in ../common
    models/        component 1 TODOs + selftest harness
    ros/           component 2 TODOs + selftest harness
    integration/   component 3 TODOs + selftest harness
    app.py         component 3 harness: the live mimic UI (fail-soft on TODOs)
    selftest.py    run all three harnesses end to end + solution parity check

## Rules of the game

- Every TODO stub raises `NotImplementedError` with a pointer to the
  notebook that teaches it. The harnesses report per-TODO status
  (PASS / FAIL / TODO / SKIP) with targeted hints.
- `_artifacts/` collects the proof: overlay images, round-trip snapshots,
  the UI screenshot.
- Two debugging lessons to keep taped to your monitor:
  - **Verify the execution provider.** Asking for the NPU isn't getting it.
  - **Mind the NPU load order.** The first model opened sizes the NPU's
    shared buffers while it stays open. Open YOLO-pose first and keep it
    open, so YOLO-detect can load after it.
