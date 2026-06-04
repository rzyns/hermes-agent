"""Tests for safe session title backfill/regeneration helpers."""

from hermes_cli.session_retitle import retitle_sessions


class FakeDB:
    def __init__(self):
        self.sessions = [
            {"id": "untitled", "title": None, "title_source": None, "source": "webui", "cwd": "/repo"},
            {"id": "manual", "title": "Manual Title", "title_source": "manual", "source": "webui"},
            {"id": "manual-untitled", "title": "Untitled", "title_source": "manual", "source": "webui"},
            {"id": "legacy-untitled", "title": "Untitled", "title_source": None, "source": "webui"},
            {"id": "auto", "title": "Generic Title", "title_source": "auto", "source": "webui"},
        ]
        self.set_calls = []

    def list_sessions_rich(self, **kwargs):
        return list(self.sessions)

    def get_messages_as_conversation(self, session_id):
        return [
            {"role": "user", "content": f"Need a better title for {session_id}"},
            {"role": "assistant", "content": "I can build a specific title."},
        ]

    def set_backfill_session_title(self, session_id, title):
        self.set_calls.append((session_id, title, "backfill"))
        return True

    def set_auto_generated_session_title(self, session_id, title):
        self.set_calls.append((session_id, title, "auto"))
        return True


def test_retitle_untitled_dry_run_reports_without_mutating():
    db = FakeDB()

    result = retitle_sessions(
        db,
        mode="untitled",
        apply=False,
        title_generator=lambda context: "Specific Untitled Session",
    )

    assert result["effect"] == "dry_run"
    assert [item["session_id"] for item in result["candidates"]] == ["untitled"]
    assert result["candidates"][0]["new_title"] == "Specific Untitled Session"
    assert result["candidates"][0]["error"] is None
    assert db.set_calls == []


def test_retitle_untitled_apply_marks_backfill_source():
    db = FakeDB()

    result = retitle_sessions(
        db,
        mode="untitled",
        apply=True,
        title_generator=lambda context: "Specific Untitled Session",
    )

    assert result["effect"] == "applied"
    assert db.set_calls == [("untitled", "Specific Untitled Session", "backfill")]


def test_retitle_auto_generated_mode_skips_manual_titles():
    db = FakeDB()

    result = retitle_sessions(
        db,
        mode="auto-generated",
        apply=True,
        title_generator=lambda context: "Specific Auto Session",
    )

    assert [item["session_id"] for item in result["candidates"]] == ["auto"]
    assert db.set_calls == [("auto", "Specific Auto Session", "auto")]


def test_retitle_apply_reports_per_item_errors_and_continues():
    class FailingDB(FakeDB):
        def __init__(self):
            super().__init__()
            self.sessions = [
                {"id": "first", "title": None, "title_source": None, "source": "webui"},
                {"id": "second", "title": None, "title_source": None, "source": "webui"},
            ]

        def set_backfill_session_title(self, session_id, title):
            if session_id == "first":
                raise ValueError("duplicate title")
            return super().set_backfill_session_title(session_id, title)

    db = FailingDB()

    result = retitle_sessions(
        db,
        mode="untitled",
        apply=True,
        title_generator=lambda context: "Generated Title",
    )

    assert [item["session_id"] for item in result["candidates"]] == ["first", "second"]
    assert result["candidates"][0]["applied"] is False
    assert result["candidates"][0]["error"] == "duplicate title"
    assert result["candidates"][1]["applied"] is True
    assert result["candidates"][1]["error"] is None
    assert db.set_calls == [("second", "Generated Title", "backfill")]
