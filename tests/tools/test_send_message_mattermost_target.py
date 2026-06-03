"""Mattermost target parsing regressions for send_message_tool."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import _parse_target_ref, send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def test_mattermost_channel_id_is_explicit():
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "mattermost",
        "i9dumu1orprwxmwgrjebbsnh5r",
    )

    assert chat_id == "i9dumu1orprwxmwgrjebbsnh5r"
    assert thread_id is None
    assert is_explicit is True


def test_resolved_mattermost_dm_name_preserves_dm_channel_id():
    mattermost_platform = Platform("mattermost")
    mattermost_cfg = SimpleNamespace(
        enabled=True,
        token="tok",
        extra={"url": "https://mattermost.example.com"},
    )
    home = SimpleNamespace(chat_id="homechannelhomechannelhomech")
    config = SimpleNamespace(
        platforms={mattermost_platform: mattermost_cfg},
        get_home_channel=lambda _platform: home,
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch(
             "gateway.channel_directory.resolve_channel_name",
             return_value="i9dumu1orprwxmwgrjebbsnh5r",
         ), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "mattermost:janusz",
                    "message": "hello",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        mattermost_platform,
        mattermost_cfg,
        "i9dumu1orprwxmwgrjebbsnh5r",
        "hello",
        thread_id=None,
        media_files=[],
        force_document=False,
    )
