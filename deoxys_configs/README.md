# NUC-side deoxys configs

These are `franka-interface` configs — the **NUC side** of deoxys. They are not
read by anything in this repo; `bin/franka-interface` reads them directly via
`YAML::LoadFile(argv[1])`, so an absolute path outside the deoxys tree is fine.

They live here so that `/home/robot-lab/work/deoxys_control` — a shared install
owned by `robot-lab` and used by other people — stays untouched. Point
`auto_arm.sh` at these instead of editing `config/franka_*.yml` in there.

```bash
cd ~/work/deoxys_control/deoxys      # cwd must stay here, see below
./auto_scripts/auto_arm.sh ~/charles/Franka-Teach/deoxys_configs/franka_left.yml
./auto_scripts/auto_arm.sh ~/charles/Franka-Teach/deoxys_configs/franka_right.yml
```

**This directory must not be named `deoxys`.** The repo root is `sys.path[0]`
for `python3 franka_server.py`, and in the `franka_teach` env the editable
install does not claim the top-level `deoxys` name via a meta-path finder — so a
`deoxys/` directory here shadows the real package as an empty namespace package
and you get `ImportError: cannot import name 'config_root' from 'deoxys'
(unknown location)`.

**The cwd still has to be the deoxys install dir.** `franka_control_node.cpp`
takes an optional second argument for the controller config and otherwise falls
back to the *relative* path `config/control_config.yml`. Only the per-arm config
moved here; that one did not.

## What differs from the shared copies

| | shared `config/franka_left.yml` | here |
|---|---|---|
| `PC.IP` | `192.168.100.83` (Lambda) | `192.168.100.201` (this NUC) |
| `POLICY_RATE` | `120` | `50` |

`franka-interface` **connects out** to `PC.IP` for its command stream, so that
value — not anything on the client side — decides which machine is allowed to
run `franka_server.py`. It is read once at startup; changing it means restarting
`franka-interface`.

`POLICY_RATE` is consumed only by the C++ side, where it sets the trajectory
interpolator's horizon (`1/policy_rate`, see
`franka-interface/include/utils/traj_interpolators/`). It has to match the rate
you actually command at — air hockey uses `control_freq=50`, so 50 here. Left at
120 the interpolator finishes each blend in 8 ms and then sits still for the rest
of the 20 ms cycle; at 20 it lags behind.

Keep these three in agreement for a given arm:

- `POLICY_RATE` here
- `POLICY_RATE` in `frankateach/configs/deoxys_<arm>_fast.yml`
- `control_freq=` passed to `franka_server.py`

The right arm's shared config (`config/franka_right.yml`) additionally pointed
`PC.IP` at `192.168.100.46`, a host that no longer answers ARP. That is why it
could not be used as-is.

Log files are written to this repo's gitignored `logs/`, again to avoid writing
into the shared tree.
