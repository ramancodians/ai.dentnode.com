"""SSRF-guarded downloader for caller-supplied mesh (STL) URLs.

This module exists because Scan Review breaks an invariant the rest of the
service holds: *"this service never dereferences a client URL."* Every other
endpoint takes bytes that the Node backend already fetched, so Node owned the
egress boundary. Scan Review takes raw file URLs, which means this process now
makes outbound requests to an address influenced by the request body — the
textbook setup for SSRF.

On Cloud Run that is not theoretical. A request to 169.254.169.254 reaches the
GCP metadata server, which will hand out a service-account access token. So the
guard here is the actual security control for the endpoint, and it is
deliberately strict:

  1. https only (http needs SCAN_REVIEW_ALLOW_INSECURE_FETCH, for local dev).
  2. Optional host allowlist (SCAN_REVIEW_ALLOWED_HOSTS) — set this in
     production so the module can only reach our own storage.
  3. Every DNS answer for the host must be a public, non-loopback,
     non-link-local, non-CGNAT address. All of them, not just the first: a
     hostname resolving to [1.2.3.4, 127.0.0.1] is rejected.
  4. The connection is **pinned to the validated IP** (Host header + SNI
     preserved) so DNS cannot be re-resolved to an internal address between the
     check and the connect. Without this, the check above is only a speed bump.
  5. Redirects are followed manually, capped, and re-validated at every hop —
     an allowed host redirecting to 169.254.169.254 is the classic bypass.
  6. Response size is capped while streaming, so a slow infinite body cannot
     exhaust memory, and Content-Length is rejected up front when it over-runs.

Nothing here trusts a header from the remote server for anything but a hint;
the file type is confirmed from the bytes.
"""

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Mesh formats we are willing to load. STL is what scanners actually export;
# the rest are here because labs occasionally receive a PLY or OBJ from a design
# suite and rejecting those outright is unhelpful.
SUPPORTED_FORMATS = ("stl", "ply", "obj", "off", "glb", "3mf")

_MAX_REDIRECTS = 3

# Enough bytes to identify every format we accept.
_SNIFF_BYTES = 128


class MeshFetchError(RuntimeError):
    """Raised when a mesh URL is unsafe, unreachable, or not a usable mesh.

    This is a *caller* error (bad or blocked URL), not an upstream model
    failure, so the endpoint records it against that one file and carries on
    with the rest rather than failing the whole request.
    """


@dataclass
class FetchedMesh:
    url: str
    label: str
    data: bytes
    file_type: str
    content_type: Optional[str] = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def _is_public_ip(ip: Any) -> bool:
    """Reject every address family that can reach infrastructure.

    `is_global` alone is not something to rely on: its semantics have shifted
    across Python patch releases (notably the 3.12.x fix for 0.0.0.0/8 and
    IPv4-mapped IPv6 handling), so the categories that matter are also checked
    explicitly. Belt and braces is the right trade here — a false negative
    blocks one scan, a false positive leaks a metadata token.
    """
    # ::ffff:169.254.169.254 must not sneak past via the IPv6 branch.
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    if isinstance(ip, ipaddress.IPv4Address):
        # Carrier-grade NAT (100.64.0.0/10) is not flagged private by
        # `ipaddress` but routes to operator infrastructure.
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return False
        # 0.0.0.0/8 — "this network"; some stacks treat 0.x as local.
        if ip in ipaddress.ip_network("0.0.0.0/8"):
            return False

    return bool(getattr(ip, "is_global", True))


def _resolve_host(host: str, port: int) -> List[Tuple[int, str]]:
    """Resolve `host` and assert every answer is publicly routable.

    Returns [(family, ip_string)] so the caller can pin the connection.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise MeshFetchError(f"Could not resolve host {host!r}: {exc}") from exc

    if settings.allow_insecure_fetch:
        # Dev escape hatch: skip the public-IP requirement so a mesh can be
        # served from localhost. Every other control (allowlist, redirect cap,
        # size cap, type sniffing) still applies.
        return [(f, s[0]) for f, _t, _p, _c, s in infos]

    resolved: List[Tuple[int, str]] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            raise MeshFetchError(f"Host {host!r} resolved to an unparseable address")

        if not _is_public_ip(ip):
            # Every answer must pass. A split-horizon record mixing a public and
            # an internal address is exactly the attack this blocks.
            raise MeshFetchError(
                f"Host {host!r} resolves to a non-public address ({ip_str}); "
                "refusing to fetch"
            )
        resolved.append((family, ip_str))

    if not resolved:
        raise MeshFetchError(f"Host {host!r} resolved to no usable address")
    return resolved


def validate_url(raw_url: str) -> Tuple[str, str, int, str]:
    """Check scheme/host policy and return (scheme, host, port, url).

    Raises MeshFetchError if the URL may not be fetched. DNS is *not* resolved
    here — that happens per redirect hop in `_fetch_once`.
    """
    url = (raw_url or "").strip()
    if not url:
        raise MeshFetchError("Empty mesh URL")

    parts = urlsplit(url)
    scheme = parts.scheme.lower()

    if scheme == "http" and not settings.allow_insecure_fetch:
        raise MeshFetchError(
            "Mesh URLs must use https "
            "(set SCAN_REVIEW_ALLOW_INSECURE_FETCH for local dev)"
        )
    if scheme not in ("https", "http"):
        raise MeshFetchError(f"Unsupported URL scheme {parts.scheme!r}")

    try:
        host = (parts.hostname or "").lower()
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise MeshFetchError(f"Malformed mesh URL: {exc}") from exc
    if not host:
        raise MeshFetchError("Mesh URL has no host")

    # Credentials in the URL are a parsing-confusion vector and no legitimate
    # scan URL needs them.
    if parts.username or parts.password:
        raise MeshFetchError("Mesh URL must not contain embedded credentials")

    allowlist = settings.allowed_hosts
    if allowlist and host not in allowlist:
        # Also honour a leading-dot suffix entry (".dentnode.com").
        if not any(h.startswith(".") and host.endswith(h) for h in allowlist):
            raise MeshFetchError(f"Host {host!r} is not in SCAN_REVIEW_ALLOWED_HOSTS")

    return scheme, host, port, url


def sniff_file_type(data: bytes, url: str, content_type: Optional[str]) -> str:
    """Identify the mesh format from the bytes, using the URL only as a hint.

    A remote server's Content-Type is not trusted; S3 serves most STLs as
    application/octet-stream anyway.
    """
    head = data[:_SNIFF_BYTES]

    # Binary STL has no reliable magic number — its 80-byte header is free text
    # and (notoriously) often literally starts with "solid", which is also the
    # ASCII STL marker. The only dependable test is the length arithmetic:
    # 80-byte header + uint32 triangle count + 50 bytes per triangle.
    if len(data) >= 84:
        n_tri = int.from_bytes(data[80:84], byteorder="little", signed=False)
        if 84 + (n_tri * 50) == len(data):
            return "stl"

    stripped = head.lstrip()
    if stripped[:5].lower() == b"solid":
        return "stl"
    if stripped[:3].lower() == b"ply":
        return "ply"
    if stripped[:3] == b"OFF":
        return "off"
    if head[:4] == b"glTF":
        return "glb"
    if head[:4] == b"PK\x03\x04":
        # Zip container — 3MF is the only zipped mesh format we accept.
        return "3mf"

    # OBJ has no magic number at all, so fall back to the extension for it.
    path = urlsplit(url).path
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in SUPPORTED_FORMATS:
        return ext

    raise MeshFetchError(
        "Downloaded file is not a recognised mesh format "
        f"(extension={ext or 'none'!r}, content-type={content_type or 'none'!r})"
    )


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """Stream the body, aborting as soon as it exceeds the cap."""
    chunks: List[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise MeshFetchError(
                f"Mesh exceeds the {max_bytes} byte limit (SCAN_REVIEW_MAX_BYTES)"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _fetch_once(
    client: httpx.AsyncClient, url: str, max_bytes: int
) -> Tuple[Optional[str], bytes, Optional[str]]:
    """One hop. Returns (redirect_location, body, content_type).

    When `redirect_location` is set the body is empty and the caller should
    re-validate that location and follow it.
    """
    scheme, host, port, url = validate_url(url)
    resolved = _resolve_host(host, port)
    _family, ip = resolved[0]

    # Pin the socket to the address just validated. Letting httpx re-resolve
    # would reopen the DNS-rebinding window, so the request goes to the literal
    # IP while Host and SNI keep the original name — virtual hosting still works
    # and TLS still verifies the certificate against `host`.
    ip_literal = f"[{ip}]" if ":" in ip else ip
    default_port = 443 if scheme == "https" else 80
    netloc = ip_literal if port == default_port else f"{ip_literal}:{port}"
    parts = urlsplit(url)
    pinned = urlunsplit((scheme, netloc, parts.path, parts.query, ""))

    host_header = host if port == default_port else f"{host}:{port}"
    request = client.build_request(
        "GET",
        pinned,
        headers={"Host": host_header, "Accept": "*/*"},
        extensions={"sni_hostname": host},
    )

    response = await client.send(request, stream=True, follow_redirects=False)
    try:
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise MeshFetchError("Redirect response had no Location header")
            # Resolve a relative redirect against the ORIGINAL url, never the
            # pinned IP url, or the next hop would inherit the IP as its host
            # and skip the allowlist check.
            return str(httpx.URL(url).join(location)), b"", None

        if response.status_code >= 400:
            raise MeshFetchError(f"Mesh URL returned HTTP {response.status_code}")

        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise MeshFetchError(
                f"Mesh is {declared} bytes, over the {max_bytes} byte limit "
                "(SCAN_REVIEW_MAX_BYTES)"
            )

        body = await _read_capped(response, max_bytes)
        return None, body, response.headers.get("content-type")
    finally:
        await response.aclose()


async def fetch_mesh(url: str, *, label: str = "scan") -> FetchedMesh:
    """Download one mesh file safely and identify its format.

    Raises MeshFetchError for anything the caller got wrong or that we refuse to
    reach.
    """
    max_bytes = settings.max_bytes
    original_url = url
    timeout = httpx.Timeout(float(settings.fetch_timeout_secs))

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            current = url
            for _hop in range(_MAX_REDIRECTS + 1):
                location, body, content_type = await _fetch_once(
                    client, current, max_bytes
                )
                if location is None:
                    if not body:
                        raise MeshFetchError("Mesh URL returned an empty body")
                    file_type = sniff_file_type(body, current, content_type)
                    return FetchedMesh(
                        url=original_url,
                        label=label,
                        data=body,
                        file_type=file_type,
                        content_type=content_type,
                    )
                current = location
            raise MeshFetchError(f"Too many redirects (> {_MAX_REDIRECTS})")
    except MeshFetchError:
        raise
    except httpx.HTTPError as exc:
        raise MeshFetchError(f"Mesh download failed: {exc}") from exc


async def fetch_all(
    files: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Optional[FetchedMesh], Optional[str]]]:
    """Fetch every requested mesh concurrently.

    Returns one (spec, mesh, error) tuple per input, in the original order. A
    failure never aborts the batch — a case with a broken upper-arch link should
    still get QA on its lower arch.
    """

    async def _one(spec: Dict[str, Any]):
        try:
            mesh = await fetch_mesh(
                spec.get("url", ""), label=spec.get("label") or "scan"
            )
            return spec, mesh, None
        except MeshFetchError as exc:
            logger.warning(
                "Mesh fetch rejected",
                extra={"label": spec.get("label"), "error": str(exc)},
            )
            return spec, None, str(exc)

    return list(await asyncio.gather(*(_one(s) for s in files)))
