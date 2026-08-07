"""Discovery-local operator and admin web application."""

import asyncio
import json
import time
from pathlib import Path

from aiohttp import WSMsgType, web

STATIC = Path(__file__).with_name("static")


def build_discovery_app(recorder, bridge, tunnel=None):
    app = web.Application()
    app["operator_ws"] = None

    async def operator(_):
        return web.FileResponse(STATIC / "operator.html")

    async def admin(_):
        return web.FileResponse(STATIC / "admin.html")

    async def status(_):
        payload = recorder.status()
        payload["tunnel_alive"] = True if tunnel is None else tunnel.alive
        return web.json_response(payload)

    async def start(request):
        body = await request.json() if request.can_read_body else {}
        try:
            task = await recorder.start(body.get("duration"))

            def consume(done):
                try:
                    done.result()
                except Exception:
                    pass

            task.add_done_callback(consume)
            return web.json_response({"ok": True, "message": "preflight started"}, status=202)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)

    async def abort(_):
        try:
            result = await recorder.abort()
            return web.json_response({"ok": True, "result": result})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)

    async def preview(_):
        if recorder.busy:
            raise web.HTTPConflict(
                text="preview is disabled during recording/finalization"
            )
        try:
            jpeg = await asyncio.to_thread(recorder.camera.snapshot, 640, 0.65)
            return web.Response(
                body=jpeg,
                content_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc

    async def admin_command(request):
        body = await request.json()
        command = body.pop("command", None)
        if not command:
            raise web.HTTPBadRequest(text="command is required")
        if recorder.busy:
            raise web.HTTPConflict(text="admin commands are disabled during an episode")
        try:
            return web.json_response(await bridge.admin(command, **body))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)

    async def websocket(request):
        current = app["operator_ws"]
        if current is not None and not current.closed:
            raise web.HTTPConflict(text="another browser already owns keyboard control")
        ws = web.WebSocketResponse(heartbeat=5)
        await ws.prepare(request)
        app["operator_ws"] = ws

        async def push():
            try:
                while not ws.closed:
                    await ws.send_json({"t": "status", **recorder.status()})
                    await asyncio.sleep(0.1)
            except (asyncio.CancelledError, ConnectionError):
                pass

        pusher = asyncio.create_task(push())
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    raw = json.loads(message.data)
                    if raw.get("t") != "keys":
                        continue
                    recorder.key_sequence += 1
                    event = {
                        "sequence": recorder.key_sequence,
                        "browser_mono_ms": raw.get("browser_mono_ms"),
                        "discovery_mono_ns": time.perf_counter_ns(),
                        "discovery_recv_mono_ns": time.perf_counter_ns(),
                        "held": sorted(set(raw.get("held") or [])),
                        "frozen": bool(raw.get("frozen", True)),
                        "home": bool(raw.get("home", False)),
                    }
                    recorder.on_keys(event)
                    await bridge.send_keys(event)
                except Exception as exc:
                    await ws.send_json({"t": "error", "message": str(exc)})
        finally:
            pusher.cancel()
            await asyncio.gather(pusher, return_exceptions=True)
            app["operator_ws"] = None
            event = {
                "sequence": recorder.key_sequence + 1,
                "browser_mono_ms": None,
                "discovery_mono_ns": time.perf_counter_ns(),
                "discovery_recv_mono_ns": time.perf_counter_ns(),
                "held": [],
                "frozen": True,
                "home": False,
            }
            recorder.key_sequence += 1
            recorder.on_keys(event)
            try:
                await bridge.send_keys(event)
            except Exception:
                pass
        return ws

    app.router.add_get("/", operator)
    app.router.add_get("/admin", admin)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/start", start)
    app.router.add_post("/api/abort", abort)
    app.router.add_get("/api/preview.jpg", preview)
    app.router.add_post("/api/admin", admin_command)
    app.router.add_get("/ws", websocket)
    return app
