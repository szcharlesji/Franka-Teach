# Air hockey runbook

Everything runs on the NUC (`robotlab-NUC8i7BEH`, 192.168.100.201) as `robot-lab`,
from `~/charles/Franka-Teach`. Have the e-stop in hand.

## 1. Desk, per arm

Tunnel from your laptop, then **include the `/desk/` path** — the redirect at `/`
drops your port:

```bash
ssh -L 8443:192.168.100.202:443 -L 8444:192.168.100.203:443 franka
```

| | Desk | robot |
|---|---|---|
| left | `https://localhost:8443/desk/` | 192.168.100.202 |
| right | `https://localhost:8444/desk/` | 192.168.100.203 |

Release brakes, enable FCI. One Desk tab at a time (Franka's limit). Click through
the cert warning.

## 2. Run it

```bash
conda activate franka_teach
cd ~/charles/Franka-Teach
python3 run.py                      # both arms; interfaces, servers, web UI
```

One process owns everything; Ctrl-C tears it all down. Then from your laptop:

```bash
ssh -L 8080:localhost:8080 franka   # open http://localhost:8080
```

Useful flags: `--arms left`, `--calibrate left`, `--speed 0.1`, `--no-interface`
(franka-interface already running in tmux), `--no-video`.

## 3. Calibrate

Mallet **off**. Press **Calibrate `<arm>`** in the page, or start with
`--calibrate <arm>`.

Jog — left `WASD` + `E`/`Q`, right `IJKL` + `O`/`U`, `Space` freezes — to each
corner **around the perimeter, not diagonally**, **Record corner** at each, then
**Finish & save**, then **Trace perimeter** and watch the real table. **Back to
play** reloads the box you just saved.

The saved box is the largest rectangle *inside* all four corners, minus `margin`
(15 mm). Check the summary: a large `corner spread` means the table is not level in
that arm's base frame — shim it, or set `plane_mode: tilted` in
`configs/airhockey.yaml` and re-teach. A large `plane residual` means it is
twisted, which no plane mode can follow.

An arm with no calibration comes up on a provisional 12 cm box around its current
pose, capped to 0.1 m/s. It will not move on startup, but it is not a play area.

## 4. Before playing for real

- `python3 scripts/rate_test.py --arm <arm> --hz 50 --dz 0.12` — both gates should
  pass. `--dz` raises the sweep off the table.
- Drive into all four walls: it should clip and hold.
- `H` homes, `Space` freezes, closing the browser tab must freeze the arm.
- **Measure that the two boxes cannot intersect.** Nothing checks this.
- Raise `speed` from 0.1 only once the above is clean. Mallet on last.

## Notes

- **`configs/airhockey.yaml` is the only guard.** It holds the coordinates a real
  arm is driven to. If you are unsure what is in it, recalibrate.
- NUC-side franka-interface configs live in `deoxys_configs/` — see the README
  there. Do not edit `~/work/deoxys_control`; it is shared.
- Stop both servers before running `tests/` — they bind the same ports, and the
  tests would otherwise drive the real arms (they now refuse instead).
- Watch the table, not the video pane; MJPEG lags the arm badly.
- The attached camera is a D435 serial `233522071078`, which is **not** in
  `configs/camera.yaml`. Add it there, or set `view_cam_id: null`.
