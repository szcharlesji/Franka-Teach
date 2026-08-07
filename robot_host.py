"""Run the NUC robot stack plus its loopback-only Discovery bridge.

This is deliberately a foreground command. It does not replace ``run.py`` and
is never installed as a boot/login service on the shared NUC.
"""

import argparse
import signal

from aiohttp import web

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.session import ArmSession, PLAY
from frankateach.airhockey.supervisor import (
    NUC_CONFIG_DIR,
    Supervisor,
    _nuc_pub_port,
    port_is_bound,
)
from frankateach.constants import HOST, arm_ports
from frankateach.recording.bridge import RobotBridge, TelemetryHub
from frankateach.recording.ownership import ArmOwnership
from frankateach.recording.profile import (
    CONTROL_HZ,
    NUM_STEPS,
    PROFILE_NAME,
    apply_recording_profile,
    validate_recording_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[PROFILE_NAME], default=PROFILE_NAME)
    parser.add_argument("--arms", default="left,right")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-interface", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if set(arms) != {"left", "right"}:
        raise SystemExit("recording_60 requires both --arms left,right")
    if args.bind not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("robot bridge must bind loopback; use an SSH tunnel")

    cfg = apply_recording_profile(ahconfig.load())
    validate_recording_profile(cfg)
    for arm in arms:
        port = arm_ports(arm)[0]
        if port_is_bound(port, HOST):
            raise SystemExit(
                f"refusing to start: {arm} control port {port} is already owned"
            )
        if not args.no_interface:
            nuc_name = cfg["arms"][arm]["nuc_config"]
            pub_port = _nuc_pub_port(NUC_CONFIG_DIR / nuc_name)
            if pub_port is not None and port_is_bound(pub_port, "0.0.0.0"):
                raise SystemExit(
                    f"refusing to start: {arm} franka-interface port {pub_port} "
                    "is already owned; stop it or use --no-interface explicitly"
                )

    ownership = ArmOwnership(arms).acquire()
    telemetry = TelemetryHub()
    supervisor = Supervisor(
        arms,
        cfg,
        control_freq=CONTROL_HZ,
        num_steps=NUM_STEPS,
        with_interface=not args.no_interface,
        with_camera=False,
    )
    sessions = {}

    def shutdown():
        for session in sessions.values():
            session.stop()
        supervisor.stop()
        ownership.release()

    try:
        supervisor.start()
        for arm in arms:
            session = ArmSession(
                arm,
                cfg,
                publish=not args.no_publish,
                telemetry_callback=telemetry.publish,
            )
            if not session.start(PLAY):
                raise RuntimeError(session.error)
            sessions[arm] = session
        bridge = RobotBridge(sessions, cfg, supervisor, telemetry)

        def sigterm(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, sigterm)
        print(f"[bridge] recording_60 ready on {args.bind}:{args.port}")
        web.run_app(
            bridge.app(),
            host=args.bind,
            port=args.port,
            handle_signals=False,
            print=None,
        )
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
