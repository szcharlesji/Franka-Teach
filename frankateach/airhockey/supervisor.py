"""Process supervision for the unified air hockey launcher.

Owns the two processes each arm needs below the Python control loop:

    auto_arm.sh <config>     franka-interface, in the deoxys install dir
    franka_server.py arm=..  the REQ/REP server our operators talk to

Both are started as child processes so a single Ctrl-C tears the whole stack
down. `auto_arm.sh` has its own restart loop, so we do not restart it ourselves;
we only report whether it is alive and whether the port it feeds came up.

Health is judged by *port*, not by process liveness: franka_server.py binds its
control port only once deoxys has answered, so a bound port is the first moment
an operator can safely connect.
"""

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from frankateach.constants import HOST, arm_ports

REPO_ROOT = Path(__file__).resolve().parents[2]
NUC_CONFIG_DIR = REPO_ROOT / "deoxys_configs"

# franka-interface reads its config from argv[1] so this can live in our repo,
# but argv[2] (the controller config) falls back to a path relative to the
# deoxys install dir -- so that has to stay the cwd. See deoxys_configs/README.md.
DEOXYS_DIR = Path(os.environ.get("DEOXYS_DIR", Path.home() / "work/deoxys_control/deoxys"))


def _nuc_pub_port(config_path):
    """NUC.PUB_PORT from a franka-interface config, or None if unreadable."""
    try:
        import yaml

        with open(config_path) as f:
            return int(yaml.safe_load(f)["NUC"]["PUB_PORT"])
    except Exception:
        return None


def port_is_bound(port, host="127.0.0.1"):
    """True if something is listening. The inverse of 'can I bind it'."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError:
        return True
    finally:
        probe.close()
    return False


class Managed:
    """One child process, with a human-readable name and a log prefix."""

    def __init__(self, name, argv, cwd=None, env=None):
        self.name = name
        self.argv = [str(a) for a in argv]
        self.cwd = str(cwd) if cwd else None
        self.env = env
        self.proc = None

    def start(self):
        print(f"[{self.name}] $ {' '.join(self.argv)}")
        self.proc = subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            env=self.env,
            stdout=None,  # inherit: children log straight to our terminal
            stderr=None,
            # Own process group, so Ctrl-C in our terminal does not race us to
            # the children -- we signal them explicitly in stop().
            start_new_session=True,
        )
        return self

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self):
        return None if self.proc is None else self.proc.poll()

    def stop(self, timeout=8.0):
        if self.proc is None or self.proc.poll() is not None:
            return
        print(f"[{self.name}] stopping (pid {self.proc.pid})")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[{self.name}] did not exit; SIGKILL")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()


class ArmStack:
    """franka-interface + franka_server for one arm."""

    def __init__(self, arm, cfg, control_freq=50, num_steps=1, with_interface=True):
        self.arm = arm
        self.control_port = arm_ports(arm)[0]
        self.with_interface = with_interface
        arm_cfg = (cfg.get("arms") or {}).get(arm) or {}
        self.deoxys_config = arm_cfg.get("deoxys_config", f"deoxys_{arm}_fast.yml")
        self.nuc_config = NUC_CONFIG_DIR / f"franka_{arm}.yml"
        self.control_freq = control_freq
        self.num_steps = num_steps
        self.interface = None
        self.server = None

    # -- lifecycle ---------------------------------------------------------
    def start_interface(self):
        if not self.with_interface:
            return
        script = DEOXYS_DIR / "auto_scripts" / "auto_arm.sh"
        if not script.exists():
            raise RuntimeError(
                f"{script} not found. Set DEOXYS_DIR to the deoxys install dir."
            )
        if not self.nuc_config.exists():
            raise RuntimeError(
                f"{self.nuc_config} not found -- see deoxys_configs/README.md"
            )
        # franka-interface binds NUC.PUB_PORT. If it is already held, another
        # instance is running for this arm -- possibly someone else's session --
        # and ours will sit in auto_arm.sh's restart loop forever. Say so once,
        # loudly, rather than letting it look like a robot fault.
        pub_port = _nuc_pub_port(self.nuc_config)
        if pub_port is not None and port_is_bound(pub_port, "0.0.0.0"):
            print(
                f"[{self.arm}/iface] WARNING: port {pub_port} is already bound, so a "
                f"franka-interface is already running for the {self.arm} arm. The "
                "instance we start cannot bind and will retry indefinitely. Kill the "
                f"existing one, or pass --no-interface to use it as-is."
            )
        self.interface = Managed(
            f"{self.arm}/iface",
            [str(script), str(self.nuc_config)],
            cwd=DEOXYS_DIR,  # argv[2] fallback is relative to here
        ).start()

    def start_server(self):
        self.server = Managed(
            f"{self.arm}/server",
            [
                shutil.which("python3") or "python3",
                "franka_server.py",
                f"arm={self.arm}",
                f"deoxys_config_path={self.deoxys_config}",
                f"control_freq={self.control_freq}",
                f"num_steps={self.num_steps}",
            ],
            cwd=REPO_ROOT,
        ).start()

    def stop_server(self):
        if self.server is not None:
            self.server.stop()

    def restart_server(self, timeout=90.0):
        """Recycle only franka_server; the interface remains running."""
        self.stop_server()
        self.start_server()
        return self.wait_for_server(timeout=timeout)

    def wait_for_server(self, timeout=90.0):
        """Block until the server answers get_state with a real pose.

        Deliberately NOT a port check. FrankaServer binds its REP socket in
        __init__, before init_server() waits for deoxys, so the port is up almost
        immediately even when franka-interface is dead -- and an operator that
        connects then blocks forever on its first request. Speaking the protocol
        is the only honest readiness signal.
        """
        import pickle

        import zmq

        deadline = time.perf_counter() + timeout
        ctx = zmq.Context.instance()
        while time.perf_counter() < deadline:
            if self.server is not None and not self.server.alive:
                raise RuntimeError(
                    f"{self.arm} franka_server exited with code "
                    f"{self.server.returncode}. Its output is above -- the usual "
                    "causes are FCI not enabled in Desk, or franka-interface not "
                    "running / pointed at a different PC.IP."
                )
            # A fresh REQ socket per attempt: REQ is strict lockstep, so a socket
            # that timed out mid-exchange cannot be reused.
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            try:
                sock.connect(f"tcp://{HOST}:{self.control_port}")
                sock.send(b"get_state")
                reply = sock.recv()
                if reply != b"state_error":
                    pickle.loads(reply)
                    return True
            except zmq.ZMQError:
                pass
            except Exception:
                pass
            finally:
                sock.close(0)
            time.sleep(0.5)
        raise RuntimeError(
            f"{self.arm} franka_server never returned a state within {timeout:.0f}s. "
            "The port is bound but deoxys is not answering: check that "
            f"franka-interface is running for the {self.arm} arm and that FCI is "
            "enabled in Desk."
        )

    def stop(self):
        # Server first: it holds the ZMQ connection to the interface.
        for m in (self.server, self.interface):
            if m is not None:
                m.stop()

    # -- reporting ---------------------------------------------------------
    def status(self):
        return {
            "arm": self.arm,
            "interface": (
                "external"
                if not self.with_interface
                else "up"
                if self.interface is not None and self.interface.alive
                else "down"
            ),
            "server": "up" if self.server is not None and self.server.alive else "down",
            "port": self.control_port,
            "port_bound": port_is_bound(self.control_port, HOST),
        }


class Supervisor:
    """All arms' process stacks, started together and torn down together."""

    def __init__(
        self,
        arms,
        cfg,
        control_freq=50,
        num_steps=1,
        with_interface=True,
        with_camera=False,
    ):
        self.stacks = {
            arm: ArmStack(
                arm,
                cfg,
                control_freq=control_freq,
                num_steps=num_steps,
                with_interface=with_interface,
            )
            for arm in arms
        }
        self.with_camera = with_camera
        self.camera = None

    def start(self, settle=2.0):
        """Start every interface, then every server, then wait for the ports.

        Interfaces first and in one pass: each takes a second or two to reach the
        robot, and starting them concurrently overlaps that.
        """
        for stack in self.stacks.values():
            stack.start_interface()
        if any(s.with_interface for s in self.stacks.values()):
            time.sleep(settle)
        for stack in self.stacks.values():
            stack.start_server()
        for stack in self.stacks.values():
            stack.wait_for_server()
            print(f"[{stack.arm}/server] port {stack.control_port} up")

        if self.with_camera:
            # Not waited on: no camera is a degraded UI, not a reason to refuse to
            # play. CameraRelay reports "no frames" and the page falls back.
            self.camera = Managed(
                "camera",
                [shutil.which("python3") or "python3", "camera_server.py"],
                cwd=REPO_ROOT,
            ).start()

    def stop(self):
        if self.camera is not None:
            self.camera.stop()
        for stack in self.stacks.values():
            stack.stop()

    def restart_server(self, arm, timeout=90.0):
        if arm not in self.stacks:
            raise ValueError(f"Unknown arm {arm!r}")
        return self.stacks[arm].restart_server(timeout=timeout)

    def status(self):
        out = {arm: stack.status() for arm, stack in self.stacks.items()}
        if self.with_camera:
            out["_camera"] = {
                "camera": "up"
                if self.camera is not None and self.camera.alive
                else "down"
            }
        return out
