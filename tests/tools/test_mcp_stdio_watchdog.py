from tools import mcp_stdio_watchdog


def test_is_orphaned_when_kernel_parent_relationship_changes():
    assert mcp_stdio_watchdog._is_orphaned(
        1234, 1.0, getppid=lambda: 999999
    ) is True


def test_is_not_orphaned_when_create_time_drifts_but_ppid_matches():
    """Regression: WSL create_time drift must not kill a live MCP server."""
    assert mcp_stdio_watchdog._is_orphaned(
        1234, 4.0, getppid=lambda: 1234
    ) is False
