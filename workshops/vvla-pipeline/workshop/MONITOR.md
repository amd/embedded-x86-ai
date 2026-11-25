# Always-on-top resource HUD

A tiny window that floats above every other window (browser, slides, terminals)
and shows just three numbers:

- **CPU %** in **burgundy** — busy % from `top`
- **GPU %** in **orange** — busy % from the amdgpu `gpu_busy_percent` sysfs gauge,
  which counts ROCm/HIP compute (the load of Gemma 4 on the iGPU), not just the graphics
  pipe `radeontop` sees; `radeontop` is used as a fallback
- **NPU inf/s** (or **Idle**) in **cyan** — inferences/sec from the change in
  `xrt-smi`'s `command_completions` counter across the NPU's hardware contexts

Every source is optional. On a plain dev machine with no `radeontop` or
`xrt-smi`, those rows read `N/A` / `Idle` and the HUD still runs.

## From the workshop folder

```bash
./launch_monitor.sh          # or:  python3 common/resource_hud.py
```

## From any notebook

The setup cell already puts `common` on the path, so:

```python
from common.monitor import open_monitor, close_monitor
open_monitor()     # pops the HUD on top of everything
close_monitor()    # dismiss it  (toggle_monitor() flips between the two)
```

## Controls while it's open

Drag anywhere to move it. Close it with its `✕`, the `Esc` key, or a
right-click. Press `t` to toggle always-on-top. Re-run the launcher (or
`open_monitor()`) to bring it back.

## Files

- `common/resource_hud.py` — the HUD (standalone; reused by both entry points).
- `common/monitor.py` — `open_monitor` / `close_monitor` / `toggle_monitor` for notebooks.
- `launch_monitor.sh` — one-line launcher for the workshop folder.
