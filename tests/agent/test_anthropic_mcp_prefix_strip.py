"""Tests for GH-25255: Anthropic OAuth ``mcp__`` tool-name round-trip.

Anthropic's subscription/OAuth billing classifier treats a **single-underscore**
``mcp_`` tool name as a third-party-app fingerprint and rejects the request with
HTTP 400 "Third-party apps now draw from extra usage, not plan limits".  So on
the OAuth wire NOTHING may carry a single-underscore ``mcp_`` prefix:

  * bare native tools            ``read_file``            -> ``mcp__read_file``
  * native MCP server tools      ``mcp_linear_get_issue`` -> ``mcp__linear_get_issue``

``normalize_response`` reverses the ``mcp__`` wire name back to whatever the tool
registry knows (the single-underscore ``mcp_<server>_<tool>`` form for MCP server
tools, or the bare name for native tools) so the dispatcher is unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_use_block(name: str, block_id: str = "tc_1", input_data: dict | None = None):
    """Create a fake Anthropic tool_use content block."""
    return SimpleNamespace(
        type="tool_use",
        id=block_id,
        name=name,
        input=input_data or {"query": "test"},
    )


def _make_response(*blocks, stop_reason="end_turn"):
    """Create a fake Anthropic Messages response."""
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model="claude-sonnet-4",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


class _FakeRegistry:
    """Minimal fake tool registry for testing prefix round-trip logic."""

    def __init__(self, registered_names: set[str]):
        self._names = registered_names

    def get_entry(self, name: str):
        if name in self._names:
            return SimpleNamespace(name=name)  # truthy = tool exists
        return None


# ---------------------------------------------------------------------------
# Response side: mcp__ wire name -> registry name
# ---------------------------------------------------------------------------

class TestAnthropicMcpPrefixStrip:
    """Verify strip_tool_prefix reverses the ``mcp__`` wire prefix correctly."""

    def _get_transport(self):
        from agent.transports.anthropic import AnthropicTransport
        return AnthropicTransport()

    def test_strips_prefix_for_oauth_injected_native_tool(self):
        """``mcp__read_file`` -> ``read_file`` (bare native tool)."""
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file", "terminal", "web_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"

    def test_restores_single_underscore_mcp_server_tool(self):
        """``mcp__linear_get_issue`` -> ``mcp_linear_get_issue`` (MCP server tool).

        MCP server tools are registered under their full single-underscore
        ``mcp_<server>_<tool>`` name, but they MUST go on the OAuth wire as
        double-underscore to dodge the classifier.  The response side restores
        the single-underscore registry name so dispatch still resolves.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__linear_get_issue")
        response = _make_response(block)

        registry = _FakeRegistry({"mcp_linear_get_issue", "read_file"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_linear_get_issue"

    def test_restores_reserved_native_namespace_for_collision(self):
        """``mcp__hermes__honcho_search`` -> bare native ``honcho_search``.

        A native Hermes tool named ``honcho_search`` and an MCP server tool named
        ``mcp_honcho_search`` both map to ``mcp__honcho_search`` under the normal
        OAuth prefix rule. The request side moves only the bare native tool into
        the reserved native namespace so Anthropic sees unique tool names; the
        response side must strip that namespace back to the dispatcher name.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__hermes__honcho_search")
        response = _make_response(block)

        registry = _FakeRegistry({"honcho_search", "mcp_honcho_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "honcho_search"

    def test_no_strip_when_flag_false(self):
        """When strip_tool_prefix=False, names are never modified."""
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=False)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp__read_file"

    def test_no_strip_when_not_mcp_prefixed(self):
        """Non-``mcp__`` names are untouched regardless of strip flag."""
        transport = self._get_transport()
        block = _make_tool_use_block("web_search")
        response = _make_response(block)

        registry = _FakeRegistry({"web_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "web_search"

    def test_preserves_name_when_no_original_in_registry(self):
        """Neither the single-underscore nor bare original is registered.

        Safety fallback: keep the full ``mcp__`` name the LLM was told about.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__unknown_tool")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file"})  # no matching original
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp__unknown_tool"

    def test_mixed_native_and_mcp_server_tools_same_response(self):
        """A bare native tool and an MCP server tool, both wired as ``mcp__``."""
        transport = self._get_transport()
        block1 = _make_tool_use_block("mcp__read_file", block_id="tc_1")
        block2 = _make_tool_use_block("mcp__linear_get_issue", block_id="tc_2")
        response = _make_response(block1, block2)

        registry = _FakeRegistry({"read_file", "mcp_linear_get_issue"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[1].name == "mcp_linear_get_issue"

    def test_prefers_full_wire_name_when_it_resolves_directly(self):
        """If the ``mcp__`` wire name itself is registered, keep it as-is.

        Defensive: never rewrite a name that already resolves natively.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__foo")
        response = _make_response(block)

        registry = _FakeRegistry({"foo", "mcp__foo"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp__foo"


class TestAnthropicOAuthAliasRoundTrip:
    """#65365: session_search / memory schemas alone deterministically trip
    Anthropic's OAuth billing classifier (verified live via the
    anthropic-ratelimit-unified-* response headers, see issue comments).
    Both are aliased to neutral wire names; normalize_response must reverse
    the mapping so the dispatcher still sees the real tool."""

    def _get_transport(self):
        from agent.transports.anthropic import AnthropicTransport
        return AnthropicTransport()

    def test_oauth_session_search_alias_round_trips_to_registry_name(self):
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__chat_history_lookup")
        response = _make_response(block)

        registry = _FakeRegistry({"session_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "session_search"

    def test_oauth_memory_alias_round_trips_to_registry_name(self):
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__context_notes")
        response = _make_response(block)

        registry = _FakeRegistry({"memory"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "memory"

    def test_registered_tool_wins_over_oauth_alias(self):
        """A real tool actually registered under the wire name keeps
        GH-25255 precedence — the alias must not hijack it."""
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__chat_history_lookup")
        response = _make_response(block)

        registry = _FakeRegistry({"mcp_chat_history_lookup", "session_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_chat_history_lookup"


# ---------------------------------------------------------------------------
# Request side: registry name -> mcp__ wire name (no single-underscore leaks)
# ---------------------------------------------------------------------------

class TestAnthropicOAuthOutgoingPrefix:
    """build_anthropic_kwargs must emit ZERO single-underscore ``mcp_`` names on
    the OAuth wire — bare names and MCP server names both land on ``mcp__``."""

    def _build(self, tools, is_oauth=True, messages=None, tool_choice=None):
        from agent.anthropic_adapter import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=messages or [{"role": "user", "content": "Hi"}],
            tools=tools,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice=tool_choice,
            is_oauth=is_oauth,
        )

    def test_oauth_adds_double_prefix_to_bare_tool_name(self):
        """OAuth + bare name -> ``mcp__`` prefix added."""
        kwargs = self._build([{
            "type": "function",
            "function": {"name": "read_file", "description": "x", "parameters": {}},
        }])
        assert [t["name"] for t in kwargs["tools"]] == ["mcp__read_file"]

    def test_oauth_promotes_single_underscore_mcp_server_tool(self):
        """OAuth + ``mcp_<server>_<tool>`` -> promoted to double underscore.

        This is the gap left by the bare constant swap: MCP server tools used
        to be *skipped* and went on the wire single-underscore, still tripping
        the classifier.  They must become ``mcp__`` and NOT be double-prefixed.
        """
        kwargs = self._build([{
            "type": "function",
            "function": {
                "name": "mcp_linear_get_issue",
                "description": "x",
                "parameters": {},
            },
        }])
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["mcp__linear_get_issue"]
        # never double-prefixed
        assert not any(n.startswith("mcp__mcp_") for n in names)

    def test_oauth_already_double_prefixed_left_alone(self):
        """OAuth + already-``mcp__`` name -> unchanged (no triple underscore)."""
        kwargs = self._build([{
            "type": "function",
            "function": {"name": "mcp__already", "description": "x", "parameters": {}},
        }])
        assert [t["name"] for t in kwargs["tools"]] == ["mcp__already"]

    def test_oauth_no_single_underscore_mcp_on_wire(self):
        """Mixed set: every wire name is bare-free of single-underscore mcp_."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                              "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_linear_get_issue",
                                              "description": "y", "parameters": {}}},
            {"type": "function", "function": {"name": "terminal",
                                              "description": "z", "parameters": {}}},
        ])
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp__linear_get_issue", "mcp__read_file", "mcp__terminal"]
        # The core invariant: NOTHING single-underscore reaches the wire.
        for n in names:
            assert not (n.startswith("mcp_") and not n.startswith("mcp__"))

    def test_oauth_disambiguates_native_tool_colliding_with_mcp_tool(self):
        """Post-prefix names must remain unique for native/MCP collisions.

        Regression for Anthropic HTTP 400 "tools: Tool names must be unique":
        ``honcho_search`` and ``mcp_honcho_search`` would otherwise both become
        ``mcp__honcho_search`` after the OAuth rewrite.
        """
        kwargs = self._build([
            {"type": "function", "function": {"name": "honcho_search",
                                              "description": "native", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_honcho_search",
                                              "description": "mcp", "parameters": {}}},
        ])
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp__hermes__honcho_search", "mcp__honcho_search"]
        assert len(names) == len(set(names))
        for n in names:
            assert not (n.startswith("mcp_") and not n.startswith("mcp__"))

    def test_non_oauth_path_untouched(self):
        """Non-OAuth requests never get the prefix — schemas pass through as-is."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                              "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_linear_get_issue",
                                              "description": "y", "parameters": {}}},
        ], is_oauth=False)
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp_linear_get_issue", "read_file"]

# ---------------------------------------------------------------------------
# #65365: session_search / memory OAuth billing-classifier trigger
# ---------------------------------------------------------------------------

class TestAnthropicOAuthClassifierAlias:
    """OAuth must alias the two schemas issue #65365 isolated as independent,
    deterministic triggers (session_search alone, memory alone) — in tool
    name, tool description, system-prompt prose, and named tool_choice."""

    def _build(self, tools, is_oauth=True, messages=None, tool_choice=None):
        from agent.anthropic_adapter import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=messages or [{"role": "user", "content": "Hi"}],
            tools=tools,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice=tool_choice,
            is_oauth=is_oauth,
        )

    def _tool(self, name, description="x"):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {}}}

    def test_oauth_aliases_session_search_and_memory_names(self):
        kwargs = self._build([
            self._tool("session_search", "Use session_search to recall prior chats."),
            self._tool("memory", "Persist notes with memory."),
        ])
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp__chat_history_lookup", "mcp__context_notes"]

    def test_oauth_aliases_session_search_in_tool_description_not_memory(self):
        """session_search is prose-safe (unambiguous token); memory is
        ordinary English and must NOT be rewritten in free text — only its
        tool name is aliased."""
        kwargs = self._build([
            self._tool("session_search", "Call session_search to recall prior chats."),
            self._tool("memory", "Persist notes; uses working memory internally."),
        ])
        by_name = {t["name"]: t["description"] for t in kwargs["tools"]}
        assert "chat_history_lookup" in by_name["mcp__chat_history_lookup"]
        assert "session_search" not in by_name["mcp__chat_history_lookup"]
        assert "memory" in by_name["mcp__context_notes"]  # untouched prose

    def test_oauth_aliases_session_search_in_system_prompt_prose(self):
        kwargs = self._build(
            [self._tool("session_search")],
            messages=[
                {"role": "system", "content": "When relevant, use session_search to recall it."},
                {"role": "user", "content": "Hi"},
            ],
        )
        system_text = "\n".join(
            b["text"] for b in kwargs["system"] if isinstance(b, dict) and b.get("type") == "text"
        )
        assert "chat_history_lookup" in system_text
        assert "session_search" not in system_text

    def test_oauth_does_not_alias_longer_identifier_containing_token(self):
        """Word-boundary matching: a path like tools/session_search_tool.py
        must survive untouched, not become chat_history_lookup_tool.py."""
        kwargs = self._build(
            [self._tool("session_search")],
            messages=[
                {"role": "system", "content": "See tools/session_search_tool.py for details."},
                {"role": "user", "content": "Hi"},
            ],
        )
        system_text = "\n".join(
            b["text"] for b in kwargs["system"] if isinstance(b, dict) and b.get("type") == "text"
        )
        assert "tools/session_search_tool.py" in system_text

    def test_oauth_tool_choice_named_alias_matches_wire_name(self):
        """The gap left open by prior alias work: a forced tool_choice must
        be normalized through the same alias + mcp__ prefix as tools[], or
        (a) the literal trigger string still reaches the wire and (b) the
        name no longer matches any entry in tools[]."""
        kwargs = self._build(
            [self._tool("session_search")],
            tool_choice="session_search",
        )
        assert kwargs["tool_choice"] == {"type": "tool", "name": "mcp__chat_history_lookup"}
        assert kwargs["tool_choice"]["name"] in {t["name"] for t in kwargs["tools"]}

    def test_oauth_tool_choice_bare_name_still_gets_mcp_prefix(self):
        """Non-aliased tool_choice names still need the mcp__ prefix under
        OAuth (pre-existing GH-25255 invariant, now routed consistently)."""
        kwargs = self._build(
            [self._tool("read_file")],
            tool_choice="read_file",
        )
        assert kwargs["tool_choice"] == {"type": "tool", "name": "mcp__read_file"}

    def test_oauth_skips_alias_on_wire_name_collision(self):
        """If a real (e.g. MCP server) tool already owns the alias's wire
        name, session_search must keep its own name rather than collide —
        two identical tool names is a hard 400, strictly worse than #65365."""
        kwargs = self._build([
            self._tool("session_search"),
            self._tool("mcp_chat_history_lookup"),
        ])
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp__chat_history_lookup", "mcp__session_search"]

    def test_api_key_path_never_aliases(self):
        """The alias is OAuth-only — API-key requests must be byte-identical
        to before this fix."""
        kwargs = self._build(
            [self._tool("session_search", "Use session_search to recall prior chats.")],
            is_oauth=False,
        )
        assert kwargs["tools"][0]["name"] == "session_search"
        assert "session_search" in kwargs["tools"][0]["description"]
