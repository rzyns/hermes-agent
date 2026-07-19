"""Focused tests for the OpenAI Images compatibility handler."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import zlib
from unittest.mock import patch

import pytest

from hermes_cli.proxy.images import (
    ImageGenerationRequest,
    _read_png_artifact,
    handle_image_generation,
)


_PAYLOAD = {
    "model": "gpt-image-2",
    "prompt": "A lighthouse",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json",
}
_ROUTE = {"provider": "openai-codex", "model": "gpt-image-2-medium"}
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c49444154789c63606060000000040001f61738550000000049454e"
    "44ae426082"
)
_TRANSPARENT_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000b49444154789c6360000200000500017a5eab3f0000000049454e44"
    "ae426082"
)


def _response_json(response):
    return json.loads(response.text)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return (
        len(data).to_bytes(4, "big")
        + chunk_type
        + data
        + crc.to_bytes(4, "big")
    )


@pytest.mark.parametrize(
    ("size", "aspect_ratio"),
    [
        ("1024x1024", "square"),
        ("1024x1536", "portrait"),
        ("1536x1024", "landscape"),
    ],
)
def test_supported_sizes_map_to_provider_aspect_ratios(size, aspect_ratio):
    request = ImageGenerationRequest.from_payload({**_PAYLOAD, "size": size})

    assert request.aspect_ratio == aspect_ratio


def test_unavailable_provider_returns_service_unavailable():
    async def run():
        with patch("agent.image_gen_registry.get_provider", return_value=None):
            return await handle_image_generation(
                _PAYLOAD,
                _ROUTE,
                asyncio.Semaphore(1),
            )

    response = asyncio.run(run())

    assert response.status == 503
    assert _response_json(response)["error"] == {
        "message": "Image provider 'openai-codex' is not available",
        "type": "server_error",
        "code": "image_provider_unavailable",
    }


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_type"),
    [
        ("invalid_argument", 400, "invalid_request_error"),
        ("auth_required", 401, "authentication_error"),
        ("capability_unsupported", 403, "permission_error"),
        ("api_error", 502, "server_error"),
    ],
)
def test_provider_failures_have_stable_http_mapping(
    error_type,
    expected_status,
    expected_type,
):
    class FailedProvider:
        def is_available(self):
            return True

        def generate(self, *args, **kwargs):
            return {
                "success": False,
                "error": "bounded provider failure",
                "error_type": error_type,
            }

    async def run():
        with patch(
            "agent.image_gen_registry.get_provider",
            return_value=FailedProvider(),
        ):
            return await handle_image_generation(
                _PAYLOAD,
                _ROUTE,
                asyncio.Semaphore(1),
            )

    response = asyncio.run(run())
    error = _response_json(response)["error"]

    assert response.status == expected_status
    assert error == {
        "message": "bounded provider failure",
        "type": expected_type,
        "code": error_type,
    }


def test_non_png_provider_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "not-an-image.png"
    artifact.write_bytes(b"not actually a PNG")

    class InvalidArtifactProvider:
        def is_available(self):
            return True

        def generate(self, *args, **kwargs):
            return {"success": True, "image": str(artifact)}

    async def run():
        with patch(
            "agent.image_gen_registry.get_provider",
            return_value=InvalidArtifactProvider(),
        ):
            return await handle_image_generation(
                _PAYLOAD,
                _ROUTE,
                asyncio.Semaphore(1),
            )

    response = asyncio.run(run())

    assert response.status == 502
    assert _response_json(response)["error"] == {
        "message": "Image provider returned an invalid PNG artifact",
        "type": "server_error",
        "code": "invalid_provider_response",
    }


def test_oversized_provider_artifact_is_rejected_before_read(tmp_path):
    artifact = tmp_path / "oversized.png"
    with artifact.open("wb") as output:
        output.write(b"\x89PNG\r\n\x1a\n")
        output.truncate(25 * 1024 * 1024 + 1)

    class OversizedArtifactProvider:
        def is_available(self):
            return True

        def generate(self, *args, **kwargs):
            return {"success": True, "image": str(artifact)}

    async def run():
        with patch(
            "agent.image_gen_registry.get_provider",
            return_value=OversizedArtifactProvider(),
        ):
            return await handle_image_generation(
                _PAYLOAD,
                _ROUTE,
                asyncio.Semaphore(1),
            )

    response = asyncio.run(run())

    assert response.status == 502
    assert _response_json(response)["error"]["code"] == "invalid_provider_response"


def test_read_png_artifact_accepts_valid_structural_png(tmp_path):
    artifact = tmp_path / "valid.png"
    artifact.write_bytes(_PNG_BYTES)

    assert _read_png_artifact(artifact) == _PNG_BYTES


def test_read_png_artifact_rejects_transparent_png(tmp_path):
    artifact = tmp_path / "transparent.png"
    artifact.write_bytes(_TRANSPARENT_PNG_BYTES)

    with pytest.raises(ValueError, match="opaque"):
        _read_png_artifact(artifact)


def test_read_png_artifact_rejects_invalid_chunk_crc(tmp_path):
    artifact = tmp_path / "bad-crc.png"
    artifact.write_bytes(_PNG_BYTES[:-1] + bytes([_PNG_BYTES[-1] ^ 0xFF]))

    with pytest.raises(ValueError, match="PNG"):
        _read_png_artifact(artifact)


def test_read_png_artifact_rejects_crc_valid_but_undecodable_png(tmp_path):
    artifact = tmp_path / "invalid-ihdr.png"
    illegal_ihdr = (
        (1).to_bytes(4, "big")
        + (1).to_bytes(4, "big")
        + bytes([0, 6, 0, 0, 0])
    )
    artifact.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", illegal_ihdr)
        + _png_chunk(b"IDAT", b"")
        + _png_chunk(b"IEND", b"")
    )

    with pytest.raises(ValueError, match="PNG"):
        _read_png_artifact(artifact)


def test_read_png_artifact_rejects_symlink(tmp_path):
    target = tmp_path / "target.png"
    target.write_bytes(_PNG_BYTES)
    artifact = tmp_path / "artifact.png"
    artifact.symlink_to(target)

    with pytest.raises((OSError, ValueError)):
        _read_png_artifact(artifact)


def test_read_png_artifact_fails_closed_without_secure_no_follow(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(_PNG_BYTES)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    def insecure_open_must_not_run(path, flags):
        raise AssertionError("artifact path must not be opened without no-follow support")

    monkeypatch.setattr(os, "open", insecure_open_must_not_run)

    with pytest.raises(OSError, match="no-follow"):
        _read_png_artifact(artifact)


def test_read_png_artifact_rejects_non_regular_file(tmp_path):
    with pytest.raises(ValueError, match="file type"):
        _read_png_artifact(tmp_path)


@pytest.mark.skipif(
    not hasattr(os, "O_NONBLOCK"),
    reason="O_NONBLOCK is required for the POSIX artifact-open regression",
)
def test_read_png_artifact_opens_non_regular_paths_nonblocking(tmp_path, monkeypatch):
    real_open = os.open

    def require_nonblocking(path, flags):
        assert flags & os.O_NONBLOCK
        return real_open(path, flags)

    monkeypatch.setattr(os, "open", require_nonblocking)

    with pytest.raises(ValueError, match="file type"):
        _read_png_artifact(tmp_path)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="FIFO regression requires POSIX mkfifo and O_NONBLOCK",
)
def test_read_png_artifact_rejects_fifo_without_blocking(tmp_path):
    artifact = tmp_path / "artifact.png"
    os.mkfifo(artifact)

    with pytest.raises(ValueError, match="file type"):
        _read_png_artifact(artifact)


def test_read_png_artifact_bounds_reads_to_validated_descriptor_size(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(_PNG_BYTES)
    real_read = os.read
    requested = []

    def bounded_read(fd, size):
        requested.append(size)
        return real_read(fd, size)

    monkeypatch.setattr(os, "read", bounded_read)

    assert _read_png_artifact(artifact) == _PNG_BYTES
    assert sum(requested) <= len(_PNG_BYTES)


def test_read_png_artifact_uses_open_descriptor_when_path_is_replaced(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(_PNG_BYTES)
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(b"not a PNG")
    real_open = os.open
    opened = False

    def open_then_replace(path, flags):
        nonlocal opened
        fd = real_open(path, flags)
        os.replace(replacement, artifact)
        opened = True
        return fd

    monkeypatch.setattr(os, "open", open_then_replace)

    assert _read_png_artifact(artifact) == _PNG_BYTES
    assert opened is True
    assert artifact.read_bytes() == b"not a PNG"


def test_read_png_artifact_rejects_file_truncated_after_fstat(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(_PNG_BYTES)
    real_fstat = os.fstat
    calls = 0

    def fstat_then_truncate(fd):
        nonlocal calls
        stat_result = real_fstat(fd)
        calls += 1
        if calls == 1:
            artifact.write_bytes(_PNG_BYTES[:20])
        return stat_result

    monkeypatch.setattr(os, "fstat", fstat_then_truncate)

    with pytest.raises(ValueError, match="size"):
        _read_png_artifact(artifact)


def test_image_generation_is_serialized_by_semaphore(tmp_path):
    artifact = tmp_path / "generated.png"
    artifact.write_bytes(_PNG_BYTES)
    active = 0
    max_active = 0

    class TrackingProvider:
        def is_available(self):
            return True

        def generate(self, *args, **kwargs):
            nonlocal active, max_active
            import time

            active += 1
            max_active = max(max_active, active)
            time.sleep(0.05)
            active -= 1
            return {"success": True, "image": str(artifact)}

    async def run():
        semaphore = asyncio.Semaphore(1)
        with patch(
            "agent.image_gen_registry.get_provider",
            return_value=TrackingProvider(),
        ):
            return await asyncio.gather(
                handle_image_generation(_PAYLOAD, _ROUTE, semaphore),
                handle_image_generation(_PAYLOAD, _ROUTE, semaphore),
            )

    responses = asyncio.run(run())

    assert [response.status for response in responses] == [200, 200]
    assert max_active == 1


def test_cancellation_keeps_semaphore_until_provider_thread_finishes(tmp_path):
    artifact = tmp_path / "generated.png"
    artifact.write_bytes(_PNG_BYTES)
    first_started = threading.Event()
    release_first = threading.Event()
    lock = threading.Lock()
    calls = 0
    active = 0
    max_active = 0

    class BlockingProvider:
        def is_available(self):
            return True

        def generate(self, *args, **kwargs):
            nonlocal calls, active, max_active
            with lock:
                calls += 1
                call_number = calls
                active += 1
                max_active = max(max_active, active)
            try:
                if call_number == 1:
                    first_started.set()
                    assert release_first.wait(timeout=2)
                return {"success": True, "image": str(artifact)}
            finally:
                with lock:
                    active -= 1

    async def run():
        semaphore = asyncio.Semaphore(1)
        with patch(
            "agent.image_gen_registry.get_provider",
            return_value=BlockingProvider(),
        ):
            first = asyncio.create_task(
                handle_image_generation(_PAYLOAD, _ROUTE, semaphore)
            )
            assert await asyncio.to_thread(first_started.wait, 1)
            first.cancel()
            second = asyncio.create_task(
                handle_image_generation(_PAYLOAD, _ROUTE, semaphore)
            )
            try:
                await asyncio.sleep(0.05)
                with lock:
                    assert calls == 1
                    assert max_active == 1
            finally:
                release_first.set()

            with pytest.raises(asyncio.CancelledError):
                await first
            response = await second
            assert response.status == 200

    asyncio.run(run())
    assert max_active == 1
