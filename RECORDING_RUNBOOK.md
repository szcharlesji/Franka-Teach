# Synchronized air-hockey recording runbook

This path records one iPhone video plus both arms' post-ramp commanded and
measured EE poses on a Discovery-controlled timeline. It is intentionally
separate from the existing 50 Hz `run.py` workflow.

## What runs where

| Device | Process |
|---|---|
| iPhone 13 Pro | CameraAPI app, foreground and in Guided Access |
| Discovery desktop | `discovery_record.py`, local browser, USB CameraAPI client, SSH tunnel, validation, and `~/data` storage |
| Robot NUC | `robot_host.py`, both Deoxys interfaces, both Franka servers, both 60 Hz operators, and the loopback bridge |
| Franka controllers | FCI and the normal low-level control stack |

The NUC never writes an episode. Discovery never connects to ports 8901/9001;
it only reaches the bridge through a managed SSH local forward. Both application
commands are foreground commands, not services.

## One-time Discovery setup

Clone both repositories without nesting one inside the other:

```bash
mkdir -p ~/charles
git clone <Franka-Teach-remote> ~/charles/Franka-Teach
git clone <Camera-API-remote> ~/Camera-API
```

Install/pair the iPhone as described by Camera-API. The recorder uses the pure
Python usbmux transport, so it does not need a persistent `iproxy`:

```bash
idevicepair pair
~/Camera-API/client/camctl --usbmux status
~/Camera-API/client/camctl --usbmux formats --min-fps 60
```

Discovery also needs `ffmpeg`/`ffprobe`; every clip is decoded and indexed before
the phone copy is eligible for deletion.

Discovery must already have an SSH alias named `franka` that reaches the robot
NUC without an interactive password prompt. The launcher adds and monitors the
port forward itself.

Edit `configs/camera_recording.yaml` on Discovery. Its null format index, lens
position, ISO, and white-balance temperature deliberately block collection. Use
`camctl --usbmux formats --min-fps 60` to select the exact 1920x1080 format
index, then
use setup preview and CameraAPI controls to find repeatable values under the
installed lights. Start near a 1/500 s shutter (`durationSeconds: 0.002`), then
tune ISO. Keep the phone fixed, powered, foregrounded, and in Guided Access.

The recording profile sets `allowFrameReordering: false` and GOP 12. The former
makes encoded packet order equal presentation order; the recorder still sorts
packet PTS defensively and rejects a file whose decode order proves that frame
reordering occurred. Do not derive `frames.csv` from decoded-frame output:
the CameraAPI contract documents that ffprobe can omit the final decoded
picture. Validation decodes the complete video for corruption detection, then
uses packet PTS and flags for the exact frame index and keyframe checks.

## Hardware gate before first collection

The four 60 Hz values are selected together by the `recording_60` profile:

- `ArmOperator.control_hz = 60`
- `FrankaServer.control_freq = 60`
- client-side `deoxys_<arm>_60.yml` has `POLICY_RATE: 60`
- NUC-side `franka_<arm>_60.yml` has `POLICY_RATE: 60`

Do not change the existing 50 Hz files. With `robot_host.py` stopped, test each
arm and then both arms at low speed. Collection is not allowed unless both
simultaneous loops sustain at least 57 Hz; there is no 50 Hz fallback.

Before every physical run, follow the Desk/FCI and calibration safety steps in
`RUNBOOK.md`. The play boxes must be calibrated, non-provisional, and physically
disjoint.

## Start a collection session

On the NUC, after releasing brakes and enabling FCI for both arms:

```bash
conda activate franka_teach
cd ~/charles/Franka-Teach
python3 robot_host.py --profile recording_60
```

This one foreground process owns both arms. It refuses a second owner, starts
the interfaces/servers/operators, and binds the bridge to NUC loopback port
8765. `Ctrl-C` freezes/parks the operators and tears down its child processes.
The normal `python3 run.py` command remains unchanged for other users.

On Discovery, with the iPhone attached over USB and CameraAPI visible in the
foreground:

```bash
conda activate franka_teach
cd ~/charles/Franka-Teach
python3 discovery_record.py \
  --role discovery \
  --session puck_demo_01 \
  --duration 20
```

Open <http://127.0.0.1:8848> directly on Discovery. The Discovery command:

- refuses to run on the robot NUC or store outside `~/data`;
- starts and monitors `ssh -NT -L 18765:127.0.0.1:8765 franka`;
- imports CameraAPI from `~/Camera-API/client` with `usbmux=True`;
- applies and reads back the camera profile;
- creates `~/data/air_hockey/puck_demo_01_<UTC>/`.

The gates need about five seconds of live robot telemetry to warm up. Start is
blocked unless camera/NUC clock uncertainty is at most 2 ms, bridge p99 RTT is at
most 20 ms, both loop rates are at least 57 Hz, both calibrations are real, the
camera configuration is exact, the phone is thermally healthy, and both devices
have the configured free-space reserve.

Discovery keeps a persistent CameraAPI SSE connection. A real
`recording.firstFrame` event with non-null `firstVideoPTSSeconds` is required;
polling `framesWritten` is not accepted as an event substitute. All server events
and their Discovery receipt timestamps are retained verbatim. `/status` is also
polled continuously, including during a clip, so a stopped/interrupted capture
session or repeated USB failure triggers an early stop and quarantine.

## Controls and episode behavior

- Left: `WASD`
- Right: `IJKL`
- `H`: home both arms
- `Space`: freeze
- `Esc`, page blur, browser disconnect, Discovery failure, or tunnel failure:
  freeze through the NUC-side watchdog

Control is continuous; Start and clip finalization never home or freeze the
arms. Start creates a fixed-duration CameraAPI recording and waits for the first
captured frame. The iPhone auto-stops. Preview is setup-only and is unavailable
while recording or finalizing.

Abort is available only during preflight/recording. It stops and deletes the
phone file, removes the local partial episode, and leaves a `manual_abort` audit
row. It is intentionally unavailable once validation/finalization begins.

The admin page at <http://127.0.0.1:8848/admin> exposes speed, arm-stack restart,
and calibration. Opening it takes keyboard ownership; every admin operation
freezes normal play first.

## Output and interpretation

Successful episodes move atomically into `raw/`; completed but invalid episodes
move into `rejected/`. Download/decode failures remain in `.partial/`, and the
phone copy is retained for retry. `camera_start.json` is written as soon as the
phone returns an ID, so the next launch reports prior partial paths and their
phone recording IDs for recovery review. A verified local raw or rejected copy
is removed from the phone.

Each episode contains the original MOV, CameraAPI configuration, SSE events and
health samples, robot/key NDJSON, raw clock samples and fits, a packet-PTS frame
index, manifest, and SHA-256 checksums. `manifest.json` declares the future
canonical action as:

```text
[left commanded box x, left commanded box y,
 right commanded box x, right commanded box y]
```

These are the absolute EE targets after acceleration ramping, box clipping, and
the measured-position leash. Keyboard states are auxiliary input records;
measured EE poses are state/diagnostics. Joint torques are produced below the
OSC interface and are not part of this raw dataset.

This recorder does not create HDF5 files and does not assign train/validation
splits.

## Hardware-free verification

Stop real Franka servers before running tests—the fake servers bind the same
ports and refuse to start if those ports are occupied.

```bash
conda activate franka_teach
python3 tests/test_box.py
python3 tests/test_teleop.py
python3 tests/test_webapp.py
python3 tests/test_session_app.py
python3 tests/test_recording.py
```

`test_recording.py` covers clock fitting, profile consistency, strict validation,
the bridge handshake/key routing/disconnect freeze, accepted and quarantined
episodes, checksums, phone cleanup ordering, and manual Abort.
