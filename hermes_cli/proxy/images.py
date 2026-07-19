"""OpenAI Images compatibility for Codex-backed proxy routes."""

from __future__ import annotations

import asyncio
import base64
import os
import stat
import time
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aiohttp import web


_SIZE_TO_ASPECT = {
    "1024x1024": "square",
    "1024x1536": "portrait",
    "1536x1024": "landscape",
}

_MAX_OUTPUT_IMAGE_BYTES = 25 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_PROVIDER_ERROR_HTTP = {
    "invalid_argument": (400, "invalid_request_error"),
    "invalid_image_input": (400, "invalid_request_error"),
    "auth_required": (401, "authentication_error"),
    "capability_unsupported": (403, "permission_error"),
    "missing_dependency": (503, "server_error"),
}


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Validated subset of ``POST /v1/images/generations`` used by Khoj."""

    model: str
    prompt: str
    aspect_ratio: str
    quality: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ImageGenerationRequest":
        model = payload.get("model")
        prompt = payload.get("prompt")
        size = payload.get("size", "1024x1024")
        quality = payload.get("quality", "auto")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required and must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required and must be a non-empty string")
        if not isinstance(size, str) or size not in _SIZE_TO_ASPECT:
            raise ValueError(
                "size must be one of 1024x1024, 1024x1536, or 1536x1024"
            )
        n = payload.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise ValueError("only n=1 is supported")
        if payload.get("response_format", "b64_json") != "b64_json":
            raise ValueError("only response_format=b64_json is supported")
        if not isinstance(quality, str) or quality not in {
            "auto",
            "low",
            "medium",
            "high",
        }:
            raise ValueError("quality must be one of auto, low, medium, or high")
        style = payload.get("style")
        if style is not None and (
            not isinstance(style, str) or style not in {"natural", "vivid"}
        ):
            raise ValueError("style must be natural or vivid when provided")
        background = payload.get("background", "auto")
        if not isinstance(background, str) or background not in {"auto", "opaque"}:
            raise ValueError("only opaque backgrounds are supported")
        if payload.get("stream", False) is not False:
            raise ValueError("streaming image generation is not supported")
        if payload.get("output_format", "png") != "png":
            raise ValueError("only output_format=png is supported")

        return cls(
            model=model.strip(),
            prompt=prompt.strip(),
            aspect_ratio=_SIZE_TO_ASPECT[size],
            quality=quality,
        )


async def handle_image_generation(
    payload: Mapping[str, Any],
    route: Mapping[str, str],
    semaphore: asyncio.Semaphore,
) -> web.Response:
    """Run a registered Hermes image provider and return OpenAI Images JSON."""

    try:
        request = ImageGenerationRequest.from_payload(payload)
    except ValueError as exc:
        return _error_response(400, str(exc), "invalid_request_error")

    from agent.image_gen_registry import get_provider

    provider_name = route.get("provider", "")
    provider = get_provider(provider_name)
    if provider is None or not provider.is_available():
        return _error_response(
            503,
            f"Image provider {provider_name!r} is not available",
            "image_provider_unavailable",
            "server_error",
        )

    provider_model = route.get("model", request.model)
    if request.quality != "auto":
        provider_model = f"gpt-image-2-{request.quality}"
    async with semaphore:
        generation_task = asyncio.create_task(
            asyncio.to_thread(
                provider.generate,
                request.prompt,
                aspect_ratio=request.aspect_ratio,
                model=provider_model,
            )
        )
        try:
            result = await asyncio.shield(generation_task)
        except asyncio.CancelledError:
            while not generation_task.done():
                try:
                    await asyncio.shield(generation_task)
                except asyncio.CancelledError:
                    continue
            with suppress(Exception):
                generation_task.result()
            raise

    if not isinstance(result, dict) or not result.get("success"):
        result = result if isinstance(result, dict) else {}
        error_code = str(result.get("error_type") or "image_generation_failed")
        status, response_type = _PROVIDER_ERROR_HTTP.get(
            error_code,
            (502, "server_error"),
        )
        message = str(result.get("error") or "Image generation failed")[:500]
        return _error_response(
            status,
            message,
            error_code,
            response_type,
        )

    image_path = Path(str(result.get("image") or ""))
    try:
        image_bytes = await asyncio.to_thread(_read_png_artifact, image_path)
    except (FileNotFoundError, OSError):
        return _error_response(
            502,
            "Image provider returned no readable image artifact",
            "invalid_provider_response",
            "server_error",
        )
    except ValueError:
        return _error_response(
            502,
            "Image provider returned an invalid PNG artifact",
            "invalid_provider_response",
            "server_error",
        )

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return web.json_response(
        {
            "created": int(time.time()),
            "data": [{"b64_json": image_b64}],
        }
    )


def _read_png_artifact(path: Path) -> bytes:
    """Read a bounded PNG artifact without exposing its path in failures."""

    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    if nofollow_flag is None:
        raise OSError("secure no-follow artifact open is unavailable")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow_flag
        | getattr(os, "O_NONBLOCK", 0)
    )
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("invalid image file type")
        if (
            before.st_size <= len(_PNG_SIGNATURE)
            or before.st_size > _MAX_OUTPUT_IMAGE_BYTES
        ):
            raise ValueError("invalid image size")

        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)

    if (
        len(raw) != before.st_size
        or after.st_size != before.st_size
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
    ):
        raise ValueError("image size changed while reading")
    _validate_png_structure(raw)
    _validate_png_decoding(raw)
    return raw


def _validate_png_structure(raw: bytes) -> None:
    """Validate PNG chunk framing and CRCs without decoding image pixels."""

    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("invalid PNG signature")

    offset = len(_PNG_SIGNATURE)
    chunk_index = 0
    saw_idat = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            raise ValueError("invalid PNG chunk framing")
        length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(raw):
            raise ValueError("invalid PNG chunk length")
        data = raw[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(raw[offset + 8 + length : chunk_end], "big")
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("invalid PNG chunk CRC")

        if chunk_index == 0:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError("invalid PNG IHDR chunk")
            if int.from_bytes(data[:4], "big") == 0 or int.from_bytes(data[4:8], "big") == 0:
                raise ValueError("invalid PNG dimensions")
        elif chunk_type == b"IHDR":
            raise ValueError("invalid duplicate PNG IHDR chunk")

        if chunk_type == b"IDAT":
            saw_idat = True
        if chunk_type == b"IEND":
            if length != 0 or not saw_idat or chunk_end != len(raw):
                raise ValueError("invalid PNG IEND chunk")
            return

        offset = chunk_end
        chunk_index += 1

    raise ValueError("invalid PNG missing IEND chunk")


def _validate_png_decoding(raw: bytes) -> None:
    """Require Pillow to recognize and fully decode a reasonably-sized PNG."""

    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ValueError("invalid PNG format")
            width, height = image.size
            if (
                width <= 0
                or height <= 0
                or max(width, height) > 8192
                or width * height > 20_000_000
            ):
                raise ValueError("invalid PNG dimensions")
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image.load()
            alpha_minimum, _ = image.convert("RGBA").getchannel("A").getextrema()
            if alpha_minimum < 255:
                raise ValueError("invalid PNG transparency; opaque output required")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid PNG image data") from exc


def _error_response(
    status: int,
    message: str,
    code: str,
    error_type: str | None = None,
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": error_type or code,
                "code": code,
            }
        },
        status=status,
    )


__all__ = ["ImageGenerationRequest", "handle_image_generation"]
