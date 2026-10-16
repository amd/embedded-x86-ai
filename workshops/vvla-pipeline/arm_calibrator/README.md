# SO-ARM101 Calibration

Calibrate an arm in one step: one pose, no joint sweeps.

## First time only

```bash
sudo apt install python3-serial
```

`calibrate.py` runs under `sudo` with the system Python. Ubuntu 24.04 refuses a
system-wide `pip install`.

## Calibrate

1. Plug the arm into USB and switch on its motor power.
2. Run:

   ```bash
   cd arm_calibrator
   sudo python3 calibrate.py
   ```

3. When prompted, move the arm to its **home pose** (all joints centered: upper
   arm vertical, forearm and gripper level, as in the photo), hold it, and
   press **Enter**.

   ![SO-101 home pose](../calibration/Image.jpg)

Done. It detects the arm, writes the calibration to the servos, verifies it,
and saves a `calibrated_<serial>.json` file next to the script. It also copies
that file to
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`,
so LeRobot, `run_pipeline.sh` and the workshop use it for
`robot_id: my_awesome_follower_arm` without asking for their own calibration.

Test the arm with the workshop's live UI, which follows you on the webcam:

```bash
cd ..                      # the vvla-pipeline folder
source activate_venv.sh    # ROS 2, .venv and the NPU runtime
cd workshop/solution
python app.py
```

---

<details>
<summary>Advanced / troubleshooting</summary>

- **Permission error?** The command uses `sudo` because it needs USB bus access.
  (Alternative: `sudo usermod -aG dialout $USER`, re-login, then run it without
  `sudo` from the repository's `.venv`, which already has `pyserial`.)
- **The bus does not answer?** The USB adapter is there, but the arm's motor
  power supply is usually off.
- **Inspect an arm without changing it:**
  ```bash
  sudo python3 read_calibration.py --port <by-id path>
  ```
- **Manual control** (custom reference/output, or dry-run without writing):
  ```bash
  # dry run (no writes)
  sudo python3 calibrate_from_reference.py --port <by-id path> \
      --ref golden_reference.json --out arm.json
  # commit to the servos
  sudo python3 calibrate_from_reference.py --port <by-id path> \
      --ref golden_reference.json --out arm.json --commit
  ```
- `golden_reference.json` is the known-good reference. Joint ranges come from it;
  each arm's homing offset is measured live from the home pose.
- LeRobot's own calibration (`python -m vla_pipeline.robot.arm_interface
  --calibrate`, see the [main README](../README.md#robot-prerequisites-hardware-runs-only))
  measures each joint's range on this arm instead.

</details>
