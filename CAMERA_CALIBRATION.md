# iPhone camera calibration for air-hockey recording

This guide produces a deterministic CameraAPI profile for synchronized
air-hockey collection. Perform it on Discovery with the iPhone mounted in its
final position and the table lighting in its final state.

The target is 1920x1080 at 60 fps, H.264, no audio, no stabilization, GOP 12,
and no frame reordering. Focus, exposure, and white balance must all be manual
in the saved profile. Automatic or merely locked values are useful for finding
a starting point, but the recorder reapplies explicit manual values so every
episode is repeatable.

## 1. Check the device and exact format

Keep the CameraAPI app foregrounded, preferably in Guided Access. On Discovery:

```bash
cd ~/Camera-API
./client/camctl --usbmux status
./client/camctl --usbmux formats --min-fps 60
```

For the currently installed iPhone 13 Pro, the selected format is:

```text
formatIndex 30
1920x1080
1-60 fps
420v
70.3 degree field of view
not binned
```

Format indices are device-specific. Re-enumerate them after changing the phone,
camera lens, or major iOS version. Do not assume that another phone's index 30
means the same thing.

## 2. Understand Python versus HTTP spelling

CameraAPI's HTTP JSON uses `whiteBalance`. Its Python client method uses the
snake-case argument `white_balance`:

```python
cam.control(
    focus={...},
    exposure={...},
    white_balance={...},
)
```

Passing `whiteBalance=` directly to `CameraAPI.control()` raises an unexpected
keyword-argument error. The Franka-Teach adapter translates the profile's HTTP
spelling to the Python spelling automatically.

## 3. Read the unrounded baseline

The compact `camctl status` display rounds several values and omits tint. Read
the complete control document:

```bash
cd ~/Camera-API
PYTHONPATH=client python3 - <<'PY'
from pprint import pprint
from camera_api import CameraAPI

cam = CameraAPI(usbmux=True)
pprint(cam.status()["controls"])
PY
```

Record the exact lens position, exposure duration, ISO, temperature, and tint.
The observed starting point on the installed rig was approximately:

```text
lens position: 0.24
exposure:      1/120 second
ISO:           113
temperature:   6435 K
```

These are starting values, not final calibration. In particular, 1/120 second
is likely to blur a fast puck.

## 4. Tune focus

Place a high-contrast target at table height near the center of the play area.
Run a single autofocus sweep and let it settle:

```bash
cd ~/Camera-API
PYTHONPATH=client python3 - <<'PY'
from pprint import pprint
from camera_api import CameraAPI

cam = CameraAPI(usbmux=True)
pprint(cam.focus_once(point=[0.5, 0.5]))
PY
```

Copy the settled `lensPosition`. Apply it manually, then check the center and
all four corners. The entire puck path matters more than maximum sharpness at
one point. If necessary, test nearby values such as 0.22, 0.24, and 0.26 and
choose the best compromise across the table.

## 5. Tune shutter and ISO

Start with a shutter near 1/480 second:

```text
durationSeconds: 0.002083333
ISO:             450
```

This has approximately the same exposure as the observed 1/120 second at ISO
113, but substantially less motion blur. Use 1/500 (`0.002`) if the lighting is
flicker-free. With mains-powered LED lighting, 1/480 or 1/240 may produce fewer
bands than 1/500.

Apply a complete candidate profile. Replace `tint` with the exact value reported
by the baseline query:

```bash
cd ~/Camera-API
PYTHONPATH=client python3 - <<'PY'
from pprint import pprint
from camera_api import CameraAPI

cam = CameraAPI(usbmux=True)
result = cam.control(
    focus={
        "mode": "manual",
        "lensPosition": 0.24,
    },
    exposure={
        "mode": "manual",
        "durationSeconds": 0.002083333,
        "iso": 450,
    },
    white_balance={
        "mode": "manual",
        "temperature": 6435,
        "tint": 0,
    },
)
pprint(result)
PY
```

Use the returned values rather than blindly copying the requested ones; the
active format may clamp shutter, ISO, temperature, or tint.

Adjust in this order:

1. Keep the shutter fixed for the desired motion sharpness.
2. Raise or lower ISO for brightness.
3. If the required ISO is excessively noisy, improve the lighting rather than
   accepting a long shutter.
4. Check for clipped white regions on the table and mallets.
5. Check for horizontal LED flicker bands.

## 6. Tune white balance

The table, puck, and mallets should have representative colors in frame. An
automatic convergence can provide a starting point:

```bash
cd ~/Camera-API
PYTHONPATH=client python3 - <<'PY'
from pprint import pprint
from camera_api import CameraAPI

cam = CameraAPI(usbmux=True)
pprint(cam.lock_everything(converge=True))
pprint(cam.status()["controls"])
PY
```

Copy the settled temperature and tint, then reapply them with
`white_balance={"mode": "manual", ...}`. Do not leave white balance automatic:
the puck or a player's clothing can cause it to change between episodes.

## 7. Inspect still images

After each change, click **AIM PREVIEW** at
<http://127.0.0.1:8848>. It loads one 640x360 JPEG. Click again to refresh.

The same snapshot can be tested directly:

```bash
curl -fsS http://127.0.0.1:8848/api/preview.jpg -o /tmp/preview.jpg
file /tmp/preview.jpg
```

Expected output includes `JPEG image data`. A still image is sufficient for
focus, brightness, clipping, and white balance, but it cannot validate puck
motion blur.

## 8. Record and inspect a motion clip

Move a puck rapidly through the play area while recording:

```bash
cd ~/Camera-API
./client/camctl --usbmux clip 5 -o /tmp/camera_tune.mov
```

Extract every 30th frame:

```bash
mkdir -p /tmp/camera_tune_frames
ffmpeg -y -i /tmp/camera_tune.mov \
  -vf "select='not(mod(n,30))'" \
  -vsync vfr \
  /tmp/camera_tune_frames/frame_%03d.jpg
```

Inspect frames containing the fastest puck motion. Reject settings with strong
motion trails, focus loss near the edges, LED bands, clipped highlights, or
unacceptable ISO noise.

## 9. Save the recorder profile

Edit `~/Franka-Teach/configs/camera_recording.yaml` with the final returned
values:

```yaml
capture:
  camera: back
  formatIndex: 30
  width: 1920
  height: 1080
  fps: 60
  codec: h264
  audio: false
  rotationDegrees: 0
  stabilization: "off"
  keyFrameInterval: 12
  allowFrameReordering: false

controls:
  focus:
    mode: manual
    lensPosition: 0.24
  exposure:
    mode: manual
    durationSeconds: 0.002083333
    iso: 450
  whiteBalance:
    mode: manual
    temperature: 6435
    tint: 0
```

The YAML deliberately uses HTTP spelling (`whiteBalance`). Franka-Teach maps it
to `white_balance` when calling the Python client.

## 10. Acceptance check

Restart `discovery_record.py` after editing the profile. Before collecting data,
require all of the following:

- Camera status reports 1920x1080 at 60 fps, H.264, audio off, stabilization
  off, rotation 0, and format index 30.
- Focus, exposure, and white balance read back at the requested manual values.
- Phone thermal state is nominal or fair.
- A moving puck is acceptably sharp throughout the play area.
- A five-second smoke recording is accepted under `raw/`, not `rejected/`.
- `captureDrops`, `writerBackpressureDrops`, and `appendFailures` are zero.
- `interruptions` is empty.
- The video contains no frame reordering and keyframe spacing is at most 12.

If the recorder blocks, use the exact preflight message rather than weakening a
gate. If a completed clip is quarantined, inspect
`manifest.json.validation.failures` and `camera.json` before changing settings.
