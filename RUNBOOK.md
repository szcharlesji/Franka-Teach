# Air hockey runbook — first robot bring-up

Everything below runs **on the NUC** (`ssh franka`) unless marked *laptop*.
Nothing here has touched a real arm yet; this is the sequence to do that safely.

Have the e-stop in hand from Step 3 onward.

## Which user runs what

`ssh franka` lands you on **`charles`**. That account owns the repo and the conda
env, and can run everything from Step 2 onward.

**Step 1 is the exception** — the deoxys config and its tmux session belong to
`robot-lab`, and `charles` has neither write access nor passwordless sudo:

| | Path | Owner |
|---|---|---|
| Repo | `/home/charles/Franka-Teach` | charles |
| Conda env | `/home/charles/envs/franka_teach` | charles |
| deoxys source + config | `/home/robot-lab/work/deoxys_control` | **robot-lab** |
| miniforge (shared, read-only) | `/home/robot-lab/miniforge3` | robot-lab |

So do Step 1 as `robot-lab` (`su - robot-lab`, or SSH in as that user), and
everything after it as `charles`.

---

## 0. Before you start

- Mallet **off** the flange for Steps 3–5. Calibration jogs near the table.
- Left arm brakes open + FCI enabled via Desk.
- Know what's inside the arm's reach. The joint reset in Step 3 moves it without
  asking.

**Reach Desk from your laptop** — no FoxyProxy needed, Desk uses relative URLs:

```bash
ssh -L 8443:192.168.100.202:443 -L 8444:192.168.100.203:443 franka
```

`https://localhost:8443` = left, `https://localhost:8444` = right. Click through
the cert warning (`CN=robot.franka.de` never matches `localhost`). Only one Desk
tab at a time — Franka's limitation, not ours.

---

## 1. Repoint franka-interface at the NUC  *(as `robot-lab`)*

`franka-interface` **connects** to `PC.IP` for its command stream. It currently
points at the Lambda, so the NUC's Python would never be heard.

```bash
su - robot-lab                 # charles cannot write this file
vim ~/work/deoxys_control/deoxys/config/franka_left.yml
```

```yaml
PC:
  IP: 192.168.100.201     # was 192.168.100.83 (Lambda)
```

While you're in there, `CONTROL.POLICY_RATE` is already `120` for left — fine for
50 Hz. The right arm's config is at `20` and will need raising to `50` when that
robot comes back.

Then restart the left arm in the existing tmux session — it belongs to
`robot-lab`, so attach as that user (`tmux attach`):

```bash
cd ~/work/deoxys_control/deoxys
./auto_scripts/auto_arm.sh config/franka_left.yml
```

Currently running, started Jul 27 — kill these first:

```
160571  bin/franka-interface config/franka_left.yml
190640  bin/gripper-interface config/franka_left.yml
```

> The gripper-interface has been pegged at 99.9% CPU for 21 h. Probably just how
> it spins, but worth a look while you're restarting.

---

## 2. Environment  *(back as `charles`)*

```bash
conda activate franka_teach
cd ~/Franka-Teach
```

The env lives at `/home/charles/envs/franka_teach` and its editable installs point
at `/home/charles/Franka-Teach`. Confirm that, because getting it wrong means
editing one copy and running another:

```bash
python -c "import frankateach; print(frankateach.__file__)"
# -> /home/charles/Franka-Teach/frankateach/__init__.py
```

If it ever prints a `/home/robot-lab/...` path, the editable install has drifted.
`deoxys` resolving to robot-lab's shared install is correct and expected.

miniforge's libmamba solver is broken on this box (`libfmt.so.10` missing); pass
`--solver=classic` to any `conda` command that needs to solve.

Sanity check with no robot involved — all three should print `FAILED: none`:

```bash
python3 tests/test_box.py
python3 tests/test_teleop.py
python3 tests/test_webapp.py
```

---

## 3. Settle the 50 Hz question  ← **first thing that moves the arm**

Terminal A:

```bash
python3 franka_server.py arm=left \
    deoxys_config_path=deoxys_left_fast.yml \
    control_freq=50 num_steps=1
```

Terminal B:

```bash
python3 scripts/rate_test.py --arm left --hz 50
```

It does a joint reset, then a ±5 cm sine in x for 20 s, and prints achieved rate
and tracking error with a pass/fail gate.

- **Both gates pass** → continue.
- **Rate gate fails** → confirm `POLICY_RATE: 50` in *both* `franka_left.yml`
  (NUC) and `deoxys_left_fast.yml` (repo), then retry. Still failing → set
  `control_hz: 30` in `configs/airhockey.yaml` and move on; nothing else changes.

---

## 4. Calibrate the left box  *(mallet off)*

```bash
python3 airhockey.py --calibrate --arm left
```

*Laptop:* `ssh -L 8080:localhost:8080 franka`, open `http://localhost:8080`.

Jog with <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>, <kbd>E</kbd>/<kbd>Q</kbd>
for height, <kbd>Space</kbd> to freeze. Click **Record corner** at each of the four
corners, **going around the perimeter** (not diagonally). Then **Finish & save**.

The saved box is the largest rectangle fitting *inside* all four corners, minus
`margin` (15 mm) — it can never reach the edge you taught. Check the summary:

- **`corner spread` > 5 mm** → the table isn't level in the arm's base frame. A
  single `plane_z` will scrape at one corner and lift at another. Re-teach more
  carefully or shim the table.
- **A "not very rectangular" warning** → your four corners were sloppy; the box
  got shrunk to stay inside them. Fine, just smaller than you may expect.

Then click **Trace perimeter** and *watch it*. This is the check that the box is
where you think it is.

---

## 5. First drive  *(mallet still off)*

Edit `configs/airhockey.yaml`:

```yaml
speed: 0.1        # start slow
```

```bash
python3 airhockey.py --arms left
```

Same tunnel, same URL. Press a movement key to take control (arms start frozen).

Check, in order:

1. Drive into all four walls — it should clip and hold, not fight.
2. Press <kbd>H</kbd> — should glide to the box centre.
3. Press <kbd>Space</kbd> — should stop dead.
4. **Close the browser tab mid-motion** — the arm must freeze. Verified in test
   as ≤6 mm of coast, but confirm it on the real thing.

Then raise `speed` until it feels right. Mallet on **last**.

---

## 6. When the right arm comes back

```bash
# NUC config/franka_right.yml: PC.IP -> 192.168.100.201, POLICY_RATE -> 50
./auto_scripts/auto_arm.sh config/franka_right.yml
python3 franka_server.py arm=right deoxys_config_path=deoxys_right_fast.yml \
    control_freq=50 num_steps=1
python3 airhockey.py --calibrate --arm right
python3 airhockey.py                      # both arms
```

**Measure that the two calibrated boxes cannot intersect.** Nothing in this stack
does cross-arm collision checking — disjoint boxes are the only thing keeping the
arms off each other.

---

## Notes

**Video.** No RealSense is attached to the NUC — `lsusb` shows only the root hubs
and Bluetooth, USB 3.0 bus empty. Until one is plugged in *here*, the page shows
"no camera publishing" and everything else works. Once attached:
`python3 camera_server.py`, and set `view_cam_id` in `configs/airhockey.yaml`.

Watch the **real table**, not the video pane — MJPEG encode + transfer + decode
lags the arm badly. Video is for awareness, not for playing.

**The config is the only guard.** `configs/airhockey.yaml` holds the coordinates
the arm is driven to. A stale or fabricated box there will drive a real arm to
fictitious coordinates. If you're unsure what's in it, re-run `--calibrate`.

**Network path.** The browser sends the full set of held keys 50×/sec rather than
key events, because over the tunnel (10.3 ms median, 29 ms p99) a single lost
`keyup` would otherwise leave an arm driving. On link loss the arm coasts at most
`watchdog` × `speed` and then freezes, with the box clip applying throughout.
