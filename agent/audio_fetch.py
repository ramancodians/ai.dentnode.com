"""SSRF-guarded downloader for Digital Ocean audio URLs.

Audio-to-Text accepts a caller-supplied Digital Ocean object URL, which makes it
the second component in this service (after scan_review) that dereferences a
client URL. This module is the security control for that endpoint and mirrors
the guard structure of ``scan_review/fetch.py``:

  1. https only (``AUDIO_TO_TEXT_ALLOW_INSECURE_FETCH`` for local dev).
  2. Host allowlist, defaulting to Digital Ocean Spaces (origin + CDN).
  3. Every DNS answer for the host must be public / non-loopback /
     non-link-local / non-CGNAT. All of them, not just the first.
  4. The connection is pinned to the validated IP (Host + SNI preserved) so DNS
     cannot be re-resolved to an internal address after the check.
  5. Redirects are followed manually, capped, and re-validated at every hop.
  6. Response size is capped while streaming.

The audio codec is confirmed from the bytes, never from headers or the URL.
"""

import ipaddress
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import settings

# Formats the audio model accepts natively. The browser MediaRecorder emits
# webm/ogg, which are NOT in this set — callers must transcode to one of these
# before upload (see the D10 upload path).
MODEL_SUPPORTED_FORMATS = ("wav", "mp3", "flac", "m4a", "mp4", "mpeg")

_MAX_REDIRECTS = 3
_SNIFF_BYTES = 12


class AudioFetchError(RuntimeError):
    """Raised when an audio URL is unsafe, unreachable, or not supported audio."""


@dataclass
class FetchedAudio:
    url: str
    data: bytes
    format: str
    content_type: Optional[str] = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def _is_public_ip(ip) -> bool:
    """Reject every address family that can reach infrastructure.

    Mirrors scan_review/fetch.py; belt-and-braces because a false negative
    blocks one file while a false positive leaks a metadata token.
    """
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
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return False
        if ip in ipaddress.ip_network("0.0.0.0/8"):
            return False
    return bool(getattr(ip, "is_global", True))


def _resolve_host(host: str, port: int):
    """Resolve `host` and assert every answer is publicly routable."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise AudioFetchError(f"Could not resolve host {host!r}: {exc}") from exc

    if settings.audio_to_text_allow_insecure_fetch:
        return [(f, s[0]) for f, _t, _p, _c, s in infos]

    resolved = []
    for family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            raise AudioFetchError(f"Host {host!r} resolved to an unparseable address")
        if not _is_public_ip(ip):
            raise AudioFetchError(
                f"Host {host!r} resolves to a non-public address ({ip_str}); refusing"
            )
        resolved.append((family, ip_str))
    if not resolved:
        raise AudioFetchError(f"Host {host!r} resolved to no usable address")
    return resolved


def validate_url(raw_url: str):
    """Check scheme/host policy and return (scheme, host, port, url)."""
    url = (raw_url or "").strip()
    if not url:
        raise AudioFetchError("Empty audio URL")

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme == "http" and not settings.audio_to_text_allow_insecure_fetch:
        raise AudioFetchError(
            "Audio URLs must use https "
            "(set AUDIO_TO_TEXT_ALLOW_INSECURE_FETCH for local dev)"
        )
    if scheme not in ("https", "http"):
        raise AudioFetchError(f"Unsupported URL scheme {parts.scheme!r}")

    try:
        host = (parts.hostname or "").lower()
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise AudioFetchError(f"Malformed audio URL: {exc}") from exc
    if not host:
        raise AudioFetchError("Audio URL has no host")

    if parts.username or parts.password:
        raise AudioFetchError("Audio URL must not contain embedded credentials")

    allowlist = settings.audio_to_text_allowed_hosts
    if allowlist and host not in allowlist:
        if not any(h.startswith(".") and host.endswith(h) for h in allowlist):
            raise AudioFetchError(
                f"Host {host!r} is not an allowed Digital Ocean host"
            )

    return scheme, host, port, url


def sniff_audio_format(data: bytes, url: str, content_type: Optional[str]) -> str:
    """Identify the audio codec from the bytes; URL/header are hints only."""
    head = data[:_SNIFF_BYTES]
    if head[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        return "m4a" if brand in (b"M4A ", b"M4B ", b"mp42", b"isom") else "mp4"

    path = urlsplit(url).path
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext:
        return ext

    raise AudioFetchError(
        "Downloaded file is not a recognised audio format "
        f"(content-type={content_type or 'none'!r})"
    )


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise AudioFetchError(
                f"Audio exceeds the {max_bytes} byte limit (AUDIO_TO_TEXT_MAX_BYTES)"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _fetch_once(client: httpx.AsyncClient, url: str, max_bytes: int):
    """One hop. Returns (redirect_location, body, content_type)."""
    scheme, host, port, url = validate_url(url)
    resolved = _resolve_host(host, port)
    _family, ip = resolved[0]

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
                raise AudioFetchError("Redirect response had no Location header")
            return str(httpx.URL(url).join(location)), b"", None
        if response.status_code >= 400:
            raise AudioFetchError(f"Audio URL returned HTTP {response.status_code}")
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise AudioFetchError(
                f"Audio is {declared} bytes, over the {max_bytes} byte limit"
            )
        body = await _read_capped(response, max_bytes)
        return None, body, response.headers.get("content-type")
    finally:
        await response.aclose()


async def fetch_audio(url: str) -> FetchedAudio:
    """Download one audio file safely and identify its codec."""
    max_bytes = settings.audio_to_text_max_bytes
    original_url = url
    timeout = httpx.Timeout(float(settings.audio_to_text_fetch_timeout_secs))

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            current = url
            for _hop in range(_MAX_REDIRECTS + 1):
                location, body, content_type = await _fetch_once(client, current, max_bytes)
                if location is None:
                    if not body:
                        raise AudioFetchError("Audio URL returned an empty body")
                    fmt = sniff_audio_format(body, current, content_type)
                    return FetchedAudio(
                        url=original_url,
                        data=body,
                        format=fmt,
                        content_type=content_type,
                    )
                current = location
            raise AudioFetchError(f"Too many redirects (> {_MAX_REDIRECTS})")
    except AudioFetchError:
        raise
    except httpx.HTTPError as exc:
        raise AudioFetchError(f"Audio download failed: {exc}") from exc
