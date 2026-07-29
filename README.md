# Franka-Teach

Bi-Manual Franka 3 robot setup.


## NUC Setup

1. Install Ubuntu 22.04 and a real-time kernel
2. Make sure the NUC is booted with the real-time kernel [[link](https://frankaemika.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel)].


## Lambda Machine Setup

todo: how to setup network, etc.


1. Setup deoxys_control. NOTE: When doing `./InstallPackage`, select `0.13.3` for installing libfranka:

```bash
git clone git@github.com:NYU-robot-learning/deoxys_control.git
mamba create -n "franka_teach" python=3.10
conda activate franka_teach
cd deoxys_control/deoxys

# Instructions from deoxys repo (this takes a while to build everything)
./InstallPackage
make -j build_deoxys=1
pip install -U -r requirements.txt
```

2. Install the Franka-Teach requirements:

```bash
cd /path/to/Franka-Teach
pip install -r requirements.txt
```

3. Install ReSkin sensor library:

```bash
git clone git@github.com:NYU-robot-learning/reskin_sensor.git
cd reskin_sensor
pip install -e .
```


## Proxy Setup

1. Install FoxyProxy extension on Chrome or Firefox. Set up the proxy like this:

![Foxy Proxy](./imgs/foxy_proxy.png)

2. Setup NUC as an ssh host like this:

```bash
Host nuc
    HostName 10.19.248.70
    User robot-lab
    LogLevel ERROR
    DynamicForward 1337
```


## How to run the Franka-Teach environment

1. Ssh into the nuc:

```bash
ssh nuc
```

2. Go to `172.16.0.4/desk` for the Franka Desk interface for the right robot and `172.16.1.4/desk` for the left robot.

The following credentials are used for the Franka Desk interface:

```
Username: GRAIL
Password: grail1234
```

NOTE: Franka Desk is cursed. You might face all sorts of issues with it. General troubleshooting:

- If it doesn't seem to connect, keep refreshing. If you lose all hope, reboot the robot.
- If end-effector doesn't show as active, you can go to the settings page and do a power off/on for the end-effector. You need to re-initialize the gripper in the same page after doing a power cycle.
- Two desk pages (for two robots) cannot be open at the same time. Close one tab and connect to the other one.

3. Open the brakes for the robot:

![open_brakes](./imgs/unlock_joints.png)

4. Enable FCI mode:

![fci](./imgs/fci.png)

5. Start the deoxys control process on the NUC:

```bash
cd /home/robot-lab/work/deoxys_control/deoxys
./auto_scripts/auto_arm.sh config/franka_left.yml # franka_right.yml for the right robot
./auto_scripts/auto_gripper.sh config/franka_left.yml # in a different terminal, if you want to use the gripper
```

6. From the Lambda, start servers:

```bash
cd /path/to/Franka-Teach/
python3 franka_server.py
python3 camera_server.py # in a different terminal
```

7. TODO: run franka_env test script:

```bash
cd /path/to/Franka-Teach/
python3 test_franka_env.py
```

## How to teleoperate

1. Do the steps until 6 in the "How to run the Franka-Teach environment" section.


2. Also, start the teleoperation script. Set the teleop mode based on if you are collecting human or robot demonstrations.:

```bash
python3 teleop.py teleop_mode=<human/robot>
```

3. You can start the data collection by running the `collect_data.py` script. Set the `demo_num` to the number of demonstrations you want to collect and `collect_depth` to `True` if you want to collect depth data from the Intel realsense cameras.

```bash
python3 collect_data.py demo_num=0 collect_depth=<True/False>
```

4. For robot teleoperation, use the VR controllers to control the robot. When collecting human data, use the VR controller to start and stop the data collection while performing the actions with the human hand.

## Air hockey (bimanual keyboard teleop)

Both arms, keyboard-driven from a **browser**, each confined to a calibrated
rectangle on a level plane at a fixed orientation. The gripper never moves —
bolt or clamp the mallet on before you start.

Everything runs on the NUC (it has no monitor, hence the web UI). You reach the
page through an SSH tunnel, exactly like the Franka Desk panels. Only key intent
crosses the network — each arm's 50 Hz control loop stays on the NUC talking to
deoxys over localhost, and freezes itself if the browser goes quiet.

### This lab's addressing

Verified 2026-07-29. The older `deoxys_left.yml` / `deoxys_right.yml` in this
repo carry a previous lab's addressing (`172.16.x.x`) and do **not** work here;
use the `_fast.yml` pair.

| | IP | deoxys PUB / SUB |
|---|---|---|
| NUC | 192.168.100.201 | — |
| Left robot | 192.168.100.202 | 5566 / 5575 |
| Right robot | 192.168.100.203 | 5556 / 5555 |
| Lambda | 192.168.100.83 | — |

### Environment

```bash
conda activate franka_teach          # ~/miniforge3/envs/franka_teach, Python 3.10
```

Note miniforge's libmamba solver is broken on this box (`libfmt.so.10` missing);
pass `--solver=classic` to any `conda` command that needs to solve.

### Reaching the robots' Desk panels from your laptop

No FoxyProxy or SOCKS proxy needed — Desk serves only relative URLs, so a plain
forward works:

```bash
ssh -L 8443:192.168.100.202:443 -L 8444:192.168.100.203:443 franka
```

Then `https://localhost:8443` (left) and `https://localhost:8444` (right).
Click through the cert warning — the cert is `CN=robot.franka.de` and will never
match `localhost`. Franka only allows one Desk tab open at a time.

### Where the Python runs

`franka-interface` on the NUC *connects* to `PC.IP` from the NUC's own
`config/franka_*.yml` for its command stream. To run the Python on the NUC, that
must be the NUC:

```yaml
PC:
  IP: 192.168.100.201     # was 192.168.100.83 (the Lambda)
```

Changing it requires restarting `franka-interface`, which disconnects anyone
driving that arm from the Lambda.

### Step 0 — confirm the arm can hold 50 Hz

Everything below assumes a 50 Hz command round trip. Verify it first.
`POLICY_RATE` must match `control_freq` in **both**
`frankateach/configs/deoxys_*_fast.yml` and the NUC-side `config/franka_*.yml`.
(The NUC's left config is already at 120; the right is at 20 and needs raising.)

```bash
python3 franka_server.py arm=left deoxys_config_path=deoxys_left_fast.yml control_freq=50 num_steps=1
python3 scripts/rate_test.py --arm left --hz 50
```

If the rate gate fails, lower `control_hz` in `configs/airhockey.yaml`; nothing
else changes.

### Step 1 — start one server per arm

```bash
python3 franka_server.py arm=left  deoxys_config_path=deoxys_left_fast.yml  control_freq=50 num_steps=1
python3 franka_server.py arm=right deoxys_config_path=deoxys_right_fast.yml control_freq=50 num_steps=1
python3 camera_server.py   # optional, for the video pane
```

Ports are arm-indexed: right uses 8900/8901/8902 (unchanged from the VR setup),
left uses 9000/9001/9002.

### Step 2 — calibrate each arm, mallet removed

```bash
python3 airhockey.py --calibrate --arm left
```

Then, from your laptop, `ssh -L 8080:localhost:8080 franka` and open
`http://localhost:8080`. Jog with <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
(+ <kbd>E</kbd>/<kbd>Q</kbd> for height) and click **Record corner** at each of the
**four corners**, going around the perimeter.

The play area is fitted as the largest rectangle that fits *inside* all four
corners, minus `margin` — so it can never reach the edge you taught. Four corners
also give the box a yaw, so <kbd>W</kbd> is "up the table" even if the arm isn't
square to it. You'll get a warning if the four corner heights disagree by more
than 5 mm, which means the table isn't level in that arm's base frame.

Then click **Trace perimeter** and watch the real limits before trusting them.

### Step 3 — play

```bash
python3 airhockey.py                 # both arms
python3 airhockey.py --arms left     # single arm
```

Tunnel and open `http://localhost:8080` as above.

| Key | Action |
|---|---|
| <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> | left arm (along the table, not the robot base axes) |
| arrows | right arm |
| <kbd>H</kbd> | glide both arms back to their box centres |
| <kbd>Space</kbd> | freeze both arms |
| <kbd>Esc</kbd> | release control (arms hold position) |

Arms start frozen; press a movement key to take control. Clicking away from the
page, closing the tab, or losing the connection freezes both arms. In every case
the arm keeps holding its pose — it does not go limp.

Tune feel in `configs/airhockey.yaml`: `speed`, `accel_time`, `max_lead`. Start
with `speed: 0.1` and raise it once you trust the boxes.

**On the network path.** The page sends the *complete set of held keys* ~50×/sec
rather than keydown/keyup events. Over the tunnel (measured 10.3 ms median,
29 ms p99), a single lost `keyup` in an edge-based protocol would leave an arm
driving into a wall; with full snapshots any loss self-heals on the next packet.
If the stream stops entirely, the arm coasts for at most `watchdog` seconds
(0.1 s → ≤35 mm at full speed, measured 6 mm) and then freezes — and the box clip
applies throughout, so it cannot leave the play area.

Watch the **real table**, not the video pane. MJPEG encode plus transfer plus
decode lags well behind the arm; the video is for awareness, not for playing.

### Recording

Each arm publishes `robot_state` and `commanded_robot_state` on its own ports, so
`collect_data.py` records unchanged. Run one collector per arm, each with its own
`storage_path` — two collectors sharing a directory will overwrite each other.

```bash
python3 collect_data.py arm=left  storage_path=/data/airhockey_left  demo_num=0
python3 collect_data.py arm=right storage_path=/data/airhockey_right demo_num=0
```

### Tests

Geometry, the control loop, and the whole web stack are covered without a robot
or a browser (they run against a fake deoxys server):

```bash
python3 tests/test_box.py       # inscribed-rectangle safety property
python3 tests/test_teleop.py    # control loop, watchdog, two-arm independence
python3 tests/test_webapp.py    # routes, key protocol, link-loss behaviour
```

### Safety

Nothing in this stack does cross-arm collision checking. The only thing keeping
the two arms apart is that their calibrated boxes are physically disjoint —
verify that by hand.
