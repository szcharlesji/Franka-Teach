"""Discovery-side client for the NUC robot bridge."""

import asyncio
import json
import time
import uuid
from collections import deque

import aiohttp

from frankateach.recording.clock import ClockSample
from frankateach.recording.protocol import PROTOCOL_VERSION


class RobotBridgeClient:
    def __init__(self, url, session_name, telemetry_callback=None, status_callback=None):
        self.url = url
        self.session_name = session_name
        self.telemetry_callback = telemetry_callback
        self.status_callback = status_callback
        self.http = None
        self.ws = None
        self.reader_task = None
        self.heartbeat_task = None
        self.connected = False
        self.error = ""
        self.status = {}
        self.rtt_ns = deque(maxlen=4000)
        self._pending_clock = {}
        self._pending_admin = {}
        self._heartbeat_sent = {}
        self._send_lock = asyncio.Lock()

    async def start(self, timeout=10.0):
        self.http = aiohttp.ClientSession()
        self.ws = await self.http.ws_connect(self.url, heartbeat=5, timeout=timeout)
        await self.ws.send_json(
            {
                "t": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "session": self.session_name,
            }
        )
        message = await self.ws.receive(timeout=timeout)
        if message.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError("robot bridge closed during hello")
        hello = json.loads(message.data)
        if hello.get("t") != "hello" or hello.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(hello.get("message") or "robot bridge protocol mismatch")
        self.connected = True
        self.reader_task = asyncio.create_task(self._reader(), name="robot-bridge-reader")
        self.heartbeat_task = asyncio.create_task(
            self._heartbeats(), name="robot-bridge-heartbeats"
        )
        return self

    async def send(self, message):
        if not self.connected or self.ws is None or self.ws.closed:
            raise RuntimeError("robot bridge is disconnected")
        async with self._send_lock:
            await self.ws.send_json(message)

    async def send_keys(self, event):
        await self.send({"t": "keys", **event})

    async def clock_samples(self, count=20, spacing=0.005):
        samples = []
        documents = []
        for _ in range(int(count)):
            probe_id = uuid.uuid4().hex
            local_send = time.perf_counter_ns()
            future = asyncio.get_running_loop().create_future()
            self._pending_clock[probe_id] = future
            await self.send(
                {
                    "t": "clock_probe",
                    "probe_id": probe_id,
                    "local_send_ns": local_send,
                }
            )
            reply, local_recv = await asyncio.wait_for(future, timeout=2.0)
            sample = ClockSample(
                local_send_ns=local_send,
                remote_recv_ns=int(reply["remote_recv_ns"]),
                remote_send_ns=int(reply["remote_send_ns"]),
                local_recv_ns=local_recv,
            )
            samples.append(sample)
            documents.append({"reply": reply, "sample": sample.to_dict()})
            await asyncio.sleep(spacing)
        return samples, documents

    async def admin(self, command, **fields):
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending_admin[request_id] = future
        await self.send(
            {"t": "admin", "request_id": request_id, "command": command, **fields}
        )
        return await asyncio.wait_for(future, timeout=120)

    async def _heartbeats(self):
        try:
            while self.connected:
                probe_id = uuid.uuid4().hex
                sent = time.perf_counter_ns()
                self._heartbeat_sent[probe_id] = sent
                await self.send(
                    {"t": "heartbeat", "probe_id": probe_id, "local_send_ns": sent}
                )
                cutoff = sent - int(10e9)
                self._heartbeat_sent = {
                    key: value for key, value in self._heartbeat_sent.items() if value >= cutoff
                }
                await asyncio.sleep(0.05)
        except (asyncio.CancelledError, RuntimeError, ConnectionError):
            pass

    async def _reader(self):
        try:
            async for message in self.ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                received = time.perf_counter_ns()
                data = json.loads(message.data)
                kind = data.get("t")
                if kind == "telemetry_batch":
                    for item in data.get("items") or []:
                        item["discovery_recv_mono_ns"] = received
                        if self.telemetry_callback is not None:
                            self.telemetry_callback(item)
                elif kind == "status":
                    self.status = data
                    if self.status_callback is not None:
                        self.status_callback(data)
                elif kind == "clock_reply":
                    future = self._pending_clock.pop(data.get("probe_id"), None)
                    if future is not None and not future.done():
                        future.set_result((data, received))
                elif kind == "heartbeat_ack":
                    sent = self._heartbeat_sent.pop(data.get("probe_id"), None)
                    if sent is not None:
                        self.rtt_ns.append(max(0, received - sent))
                elif kind == "admin_result":
                    future = self._pending_admin.pop(data.get("request_id"), None)
                    if future is not None and not future.done():
                        future.set_result(data)
                elif kind == "error":
                    request_id = data.get("request_id")
                    future = self._pending_admin.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_exception(RuntimeError(data.get("message", "bridge error")))
                    else:
                        self.error = data.get("message", "bridge error")
        except (asyncio.CancelledError, ConnectionError, aiohttp.ClientError) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self.error = str(exc)
        finally:
            self.connected = False
            error = RuntimeError(self.error or "robot bridge disconnected")
            for pending in (self._pending_clock, self._pending_admin):
                for future in pending.values():
                    if not future.done():
                        future.set_exception(error)
                pending.clear()

    async def close(self):
        self.connected = False
        for task in (self.heartbeat_task, self.reader_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self.heartbeat_task, self.reader_task) if task is not None),
            return_exceptions=True,
        )
        if self.ws is not None:
            await self.ws.close()
        if self.http is not None:
            await self.http.close()
