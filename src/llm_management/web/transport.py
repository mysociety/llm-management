"""
HTTP transport restricted to validated public destinations.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpx2


async def public_address(host: str, port: int) -> str:
    """
    Resolve once and reject every non-public destination before connecting.
    """
    addresses = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    )
    ips = [item[4][0] for item in addresses]
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError("Page destination must use public IP addresses")
    return ips[0]


class PublicWebTransport(httpx2.AsyncBaseTransport):
    """
    Pin connections to validated IPs, including every redirect target.

    Preserve HTTP Host and TLS SNI/certificate checks for the original hostname.
    This avoids a second DNS lookup and does not use environment proxies.
    """

    def __init__(self) -> None:
        # Pool origins use the pinned IP. Do not reuse a TLS connection for
        # another hostname that happens to resolve to the same shared IP.
        self.transport: httpx2.AsyncBaseTransport = httpx2.AsyncHTTPTransport(
            trust_env=False,
            retries=0,
            limits=httpx2.Limits(max_connections=10, max_keepalive_connections=0),
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        url = request.url
        if (
            url.scheme not in {"http", "https"}
            or url.username
            or url.password
            or url.port not in {None, 80, 443}
        ):
            raise ValueError(
                "Only public HTTP(S) pages on standard ports are supported"
            )
        host = url.host
        ip = await public_address(
            host, url.port or (443 if url.scheme == "https" else 80)
        )
        headers = request.headers.copy()
        headers["Host"] = url.netloc.decode("ascii")
        pinned = httpx2.Request(
            request.method,
            url.copy_with(host=ip),
            headers=headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": host},
        )
        return await self.transport.handle_async_request(pinned)

    async def aclose(self) -> None:
        await self.transport.aclose()
