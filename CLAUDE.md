# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Bi-manual Franka Panda teleoperation. Two independent stacks share one robot layer:

- **VR teleop** (original) — Oculus controller → `teleop.py` → `FrankaOperator`
- **Air hockey** (`airhockey.py`) — browser keyboard → `ArmOperator`, arms confined to calibrated rectangles on a level plane

## Commands

```bash
conda activate franka_teach       # deoxys is not on PyPI; it must be pip install -e'd from source

# Tests — plain scripts, not pytest. Each prints PASS/FAIL lines then "FAILED: none".
python3 tests/test_box.py         # box geometry; pure numpy, no robot
python3 tests/test_teleop.py      # control loop vs tests/fake_server.py
python3 tests/test_webapp.py      # HTTP routes, key protocol, link-loss behaviour

# There is no test runner and no way to select a single case — edit the
# __main__ block at the bottom of a file to skip test functions.

pre-commit run --all-files        # ruff lint (--fix) + ruff format

# Robot stack (one server per arm; see RUNBOOK.md before running any of this)
python3 franka_server.py arm=left deoxys_config_path=deoxys_left_fast.yml control_freq=50 num_steps=1
python3 camera_server.py
python3 scripts/rate_test.py --arm left --hz 50   # gate: can the arm sustain the command rate?

python3 airhockey.py --calibrate --arm left       # teach the play box (browser UI)
python3 airhockey.py --verify --arm left          # trace the calibrated perimeter
python3 airhockey.py --arms left                  # play
```

`franka_server.py`, `camera_server.py`, `collect_data.py`, `teleop.py`, `reskin_server.py` use **hydra** (`configs/*.yaml`, override as `key=value`). `airhockey.py` uses **argparse** plus a hand-parsed `configs/airhockey.yaml` — calibration rewrites one arm's block in place, which OmegaConf would not round-trip.

## Architecture

### Control path

```
input source ──▶ ArmOperator/FrankaOperator ──REQ/REP──▶ FrankaServer ──▶ deoxys ──▶ arm
                 (absolute EE pose @ N Hz)     ZMQ        (osc_move)      (NUC)
```

`FrankaServer` (`frankateach/franka_server.py`) is a synchronous REQ/REP loop: one request in, one `FrankaState` out. Its interface is *"here is an absolute EE pose"* — nothing about it is VR- or keyboard-specific, which is why both stacks reuse it unchanged. `Robot` subclasses deoxys' `FrankaInterface` and issues `num_steps` OSC_POSE deltas per request.

**`ArmOperator.set_intent()` is the input seam.** Everything safety-relevant (box clip, leash, watchdog) lives inside the operator's own thread, so a hung, crashed, or *disconnected* UI freezes the arm instead of running it. Swapping input sources means writing to that seam — do not add safety logic on the input side.

### Ports are arm-indexed

`arm_ports(arm)` in `constants.py` returns `(control, state, commanded)`. Right keeps the historical `8901/8900/8902`; left is offset by 100. `CONTROL_PORT` etc. still exist as the right-arm defaults so the VR path and `collect_data.py` are unaffected. Anything that binds or connects one of these must go through `arm_ports()`.

### Air hockey specifics (`frankateach/airhockey/`)

- `box.py` — `PlayBox`, an oriented rectangle. `box_from_corners()` fits the **inscribed** rectangle to four taught corners (`hx = min(|x'|)` in box frame, minus `margin`), so the playable area is always strictly inside what was taught. This is the only thing keeping the mallet off the table edge; `tests/test_box.py` asserts it over randomly skewed quads. Four corners also yield a yaw, so keys move along *table* axes rather than robot base axes.
- `operator.py` — `ArmOperator` (2-DoF play loop, z and orientation pinned) and `JogOperator` (3-DoF, calibration).
- `webapp.py` + `static/index.html` — aiohttp server: page, MJPEG video, websocket. Serves both play and calibrate modes.
- `control.py` — `ArmLink` (REQ socket + absolute-pose send), `RateLimiter`, `ramp`.

The UI is a **browser page, not a desktop window**, because the deployment NUC has no monitor. There is no pygame anywhere.

### Two non-obvious invariants

**Do not name a `threading.Event` `self._stop` on a `Thread` subclass.** It shadows `threading.Thread._stop()`, which `join()` calls during teardown; the result is a `TypeError` on every clean shutdown. Use `_stop_event`. This bit `ArmOperator` and `CameraFeed` once already and only reproduced on Linux.

**Do not use `frankateach.utils.FrequencyTimer` in the air hockey loops.** Its `end_loop()` busy-waits, holding the GIL and starving the other arm's thread. `control.RateLimiter` sleeps instead, and additionally compensates for `time.sleep` overshoot (measured ~4.7 ms at a 20 ms request, ~9 ms with a busy sibling thread) — without that compensation a nominal 50 Hz loop actually runs at ~43 Hz.

### Testing without hardware

`tests/fake_server.py` speaks the same REQ/REP protocol as `FrankaServer` and models the arm as a first-order system chasing the command. It makes the entire stack testable except deoxys itself — clipping, watchdog, rate, two-arm thread independence, websocket behaviour. Prefer extending it over mocking.

Note `tests/test_teleop.py` must `join()` each `FakeServer` before the next test binds the same port.

## Deployment reality (see RUNBOOK.md)

- **The repo's `deoxys_left.yml` / `deoxys_right.yml` are stale** — they carry a previous lab's `172.16.x.x` addressing that does not exist on the current network. Use the `_fast.yml` pair, which was verified against the NUC's own configs.
- **`franka-interface` connects *out* to `PC.IP`** from the NUC-side `config/franka_*.yml`. That value, not anything in this repo, decides which machine may run the Python. Changing it requires restarting `franka-interface`.
- `POLICY_RATE` must match `control_freq` in **both** the client `_fast.yml` and the NUC-side config.

## Safety

`configs/airhockey.yaml` holds the coordinates a real arm is driven to. A stale or fabricated box there will drive the arm to fictitious positions — test code that calls `ahconfig.save_box()` must not run against the real config file.

Nothing in this stack does cross-arm collision checking. Disjoint calibrated boxes are the only thing keeping the two arms apart.
