"""Run the synchronized iPhone + bimanual air-hockey recorder on Discovery."""

import argparse
import asyncio
import signal
from pathlib import Path

from aiohttp import web

from frankateach.recording.camera import CameraAPIAdapter
from frankateach.recording.recorder import EpisodeRecorder
from frankateach.recording.robot_client import RobotBridgeClient
from frankateach.recording.storage import SessionStore
from frankateach.recording.tunnel import SSHTunnel
from frankateach.recording.web import build_discovery_app

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["discovery"], required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--storage-root", default="~/data")
    parser.add_argument("--camera-api-root", default="~/Camera-API")
    parser.add_argument(
        "--camera-profile", default=str(REPO_ROOT / "configs" / "camera_recording.yaml")
    )
    parser.add_argument("--ssh-host", default="franka")
    parser.add_argument("--bridge-local-port", type=int, default=18765)
    parser.add_argument("--bridge-remote-port", type=int, default=8765)
    parser.add_argument("--no-tunnel", action="store_true", help="testing only")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bind not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Discovery UI must bind loopback")

    store = SessionStore(
        args.storage_root,
        args.session,
        enforce_discovery=True,
    )
    tunnel = None
    bridge = None
    camera = None
    camera_health_task = None
    try:
        if not args.no_tunnel:
            tunnel = SSHTunnel(
                args.ssh_host, args.bridge_local_port, args.bridge_remote_port
            ).start()
        camera = CameraAPIAdapter.from_repo(args.camera_api_root, usbmux=True)
        bridge = RobotBridgeClient(
            f"http://127.0.0.1:{args.bridge_local_port}/ws", store.label
        )
        recorder = EpisodeRecorder(
            store,
            camera,
            bridge,
            args.camera_profile,
            REPO_ROOT,
            args.camera_api_root,
            default_duration=args.duration,
        )
        bridge.telemetry_callback = recorder.on_telemetry
        app = build_discovery_app(recorder, bridge, tunnel=tunnel)

        async def startup(_):
            nonlocal camera_health_task
            await bridge.start()
            try:
                await recorder.prepare_camera()
                camera_health_task = asyncio.create_task(
                    recorder.camera_health_loop(),
                    name="cameraapi-health",
                )
                recorder.state = "idle"
                recorder.message = (
                    "camera and robot bridge ready; waiting for strict gate warm-up"
                )
                if recorder.orphan_partials:
                    recorder.message += (
                        f"; {len(recorder.orphan_partials)} prior partial episode(s) "
                        "need recovery review"
                    )
            except Exception as exc:
                recorder.state = "blocked"
                recorder.message = str(exc)

        async def cleanup(_):
            if camera_health_task is not None:
                camera_health_task.cancel()
                await asyncio.gather(camera_health_task, return_exceptions=True)
            if bridge is not None:
                await bridge.close()
            if camera is not None:
                await asyncio.to_thread(camera.close)
            store.close()
            if tunnel is not None:
                tunnel.stop()

        app.on_startup.append(startup)
        app.on_cleanup.append(cleanup)

        def sigterm(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, sigterm)
        print(f"Discovery recorder: http://{args.bind}:{args.port}")
        print(f"Raw session: {store.path}")
        web.run_app(
            app,
            host=args.bind,
            port=args.port,
            handle_signals=False,
            print=None,
        )
    except KeyboardInterrupt:
        pass
    finally:
        store.close()
        if camera is not None:
            camera.close()
        if tunnel is not None:
            tunnel.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
