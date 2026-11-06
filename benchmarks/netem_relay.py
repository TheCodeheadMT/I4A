"""Loopback TCP relay that adds round-trip delay and an optional bandwidth cap (whitepaper 6.2 Stage 5).

It stands in for an internet path between the coordinator and an RPC worker without changing any system
network settings. Each direction is delayed by half the requested round trip; bytes keep their order.
Python scheduling adds roughly a millisecond of its own, so always compare against a relay run at 0 ms.

Usage: python benchmarks/netem_relay.py --listen 60052 --target 50052 --rtt-ms 80 [--mbps 50]
"""
import argparse
import asyncio
import socket


class DelayRelay:
    def __init__(self, listen_port: int, target_port: int, rtt_ms: float, mbps: float | None = None):
        self.listen_port, self.target_port = listen_port, target_port
        self.one_way = rtt_ms / 2000.0
        self.bytes_per_s = mbps * 1e6 / 8 if mbps else None
        self.server: asyncio.AbstractServer | None = None

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        link_free = 0.0

        async def receive():
            nonlocal link_free
            while True:
                data = await reader.read(1 << 16)
                now = loop.time()
                if self.bytes_per_s and data:  # serialisation delay on a capped link
                    link_free = max(link_free, now) + len(data) / self.bytes_per_s
                    due = link_free + self.one_way
                else:
                    due = now + self.one_way
                queue.put_nowait((due, data))
                if not data:
                    return

        async def send():
            while True:
                due, data = await queue.get()
                wait = due - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                if not data:
                    writer.close()
                    return
                writer.write(data)
                await writer.drain()

        try:
            await asyncio.gather(receive(), send())
        except (ConnectionError, asyncio.CancelledError):
            writer.close()

    async def _handle(self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        try:
            up_r, up_w = await asyncio.open_connection("127.0.0.1", self.target_port)
        except OSError:
            client_w.close()
            return
        for w in (client_w, up_w):
            sock = w.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        await asyncio.gather(self._pipe(client_r, up_w), self._pipe(up_r, client_w), return_exceptions=True)

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", self.listen_port)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def _main(a) -> None:
    relay = DelayRelay(a.listen, a.target, a.rtt_ms, a.mbps)
    await relay.start()
    print(f"relay 127.0.0.1:{a.listen} -> 127.0.0.1:{a.target} rtt={a.rtt_ms}ms mbps={a.mbps or 'unlimited'}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--rtt-ms", type=float, default=0.0)
    ap.add_argument("--mbps", type=float, default=None)
    asyncio.run(_main(ap.parse_args()))
