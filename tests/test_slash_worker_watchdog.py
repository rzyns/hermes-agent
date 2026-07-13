from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_ignores_create_time_drift_while_parent_relationship_is_live():
    # WSL can shift psutil's epoch-derived create_time for a live process.
    # The kernel parent relationship remains authoritative.
    assert slash_worker._is_orphaned(1234, 0.0, getppid=lambda: 1234) is False


def test_is_orphaned_false_when_parent_alive_and_matches():
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 1234) is False
