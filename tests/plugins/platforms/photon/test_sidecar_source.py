"""Regression checks for the Photon Node sidecar source.

These are intentionally source-level: the sidecar is an ESM executable that starts
an HTTP server at module load, so importing private helpers directly would bind
ports and/or contact Spectrum. The checks below guard the exact SDK-compatibility
contract that broke cold outbound sends with spectrum-ts 3.1.x.
"""
from __future__ import annotations

from pathlib import Path


SIDECAR = (
    Path(__file__).resolve().parents[4]
    / "plugins/platforms/photon/sidecar/index.mjs"
)


def _source() -> str:
    return SIDECAR.read_text(encoding="utf-8")


def test_sidecar_supports_function_style_spectrum_space_api() -> None:
    source = _source()

    # spectrum-ts 3.1.x exposes `im.space` as a callable function, not an
    # object with only `.create()` / `.get()` methods.
    assert 'typeof im.space === "function"' in source
    assert "im.space(phoneTarget)" in source

    # Older SDK shapes are still supported, but guarded by capability checks so
    # a missing method cannot raise TypeError and become a generic HTTP 500.
    assert 'typeof im.space?.create === "function"' in source
    assert 'typeof im.space?.get === "function"' in source


def test_sidecar_caches_inbound_spaces_under_phone_and_sender_ids() -> None:
    source = _source()

    # Inbound DM spaces may have opaque ids while the usable delivery target is
    # carried separately. Cache all stable identifiers before trying cold
    # outbound resolution again.
    assert "space?.phone" in source
    assert "msgSpace.phone" in source
    assert "sender.id" in source
