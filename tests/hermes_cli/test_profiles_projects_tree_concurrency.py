"""HTTP bursts must not occupy the worker pool used by unrelated sync routes."""

import asyncio
import threading

import anyio
import httpx
import pytest
from fastapi import FastAPI

from hermes_cli.web_routers import profiles

TREE = "/api/profiles/projects/tree"


def _app():
    app = FastAPI()
    app.include_router(profiles.sessions_router)

    @app.get("/probe")
    def probe():
        return {"ok": True}

    return app


@pytest.mark.parametrize("distinct", [False, True], ids=["identical", "different-parameters"])
def test_tree_burst_leaves_sync_http_responsive(monkeypatch, distinct):
    app = _app()

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls = []

        def targets(*args, **kwargs):
            calls.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            assert release.wait(10), "test did not release tree enumeration"
            return []

        monkeypatch.setattr(profiles, "_profile_targets", targets)
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = 2
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            requests = [asyncio.create_task(client.get(
                TREE, params={"preview_limit": i if distinct else 3})) for i in range(8)]
            try:
                await asyncio.wait_for(started.wait(), 5)
                response = await asyncio.wait_for(client.get("/probe"), 2)
                assert response.json() == {"ok": True}
                assert limiter.borrowed_tokens == 0
                assert 1 <= len(calls) <= 2  # bounded even for distinct keys
            finally:
                release.set()
                responses = await asyncio.gather(*requests)
                limiter.total_tokens = previous
            assert all(r.status_code == 200 for r in responses)
            assert all(r.json() == responses[0].json() for r in responses)
            expected = len(requests) if distinct else 1
            assert len(calls) == expected
            # Single-flight, not a persistent response cache.
            assert (await client.get(TREE)).status_code == 200
            assert len(calls) == expected + 1

    # Reuse the same app across fresh loops (as non-context-managed TestClient does).
    asyncio.run(run())
    asyncio.run(run())


@pytest.mark.parametrize("explicit_home", [False, True], ids=["fallback-home", "cold-root"])
def test_tree_scope_filesystem_work_stays_off_event_loop(monkeypatch, tmp_path, explicit_home):
    from pathlib import Path
    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    if explicit_home:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-root"))
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)
    native_home = hermes_constants._get_platform_default_hermes_home()
    native_home.mkdir(parents=True)
    (native_home / "active_profile").write_text("default", encoding="utf-8")
    app = _app()

    def build(*args):
        return {"home": str(hermes_constants.get_hermes_home()),
                "root": str(hermes_constants.get_default_hermes_root())}

    monkeypatch.setattr(profiles, "_build_profiles_projects_tree", build)

    async def run():
        loop_thread = threading.get_ident()
        calls = []

        def track(name, original):
            def wrapped(self, *args, **kwargs):
                calls.append((name, threading.get_ident()))
                return original(self, *args, **kwargs)
            return wrapped

        with monkeypatch.context() as patch:
            for name in ("resolve", "exists", "read_text"):
                patch.setattr(Path, name, track(name, getattr(Path, name)))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                        base_url="http://test") as client:
                response = await client.get(TREE)
        assert response.status_code == 200
        assert calls, "exercise actual scope filesystem resolution, not a mocked resolver"
        assert not [name for name, thread in calls if thread == loop_thread]

    token = hermes_constants.set_hermes_home_override(None)
    try:
        asyncio.run(run())
    finally:
        hermes_constants.reset_hermes_home_override(token)


async def _request(client, **kwargs):
    """Return only after the request has run to its first async suspension."""
    entered = asyncio.Event()

    async def get():
        asyncio.get_running_loop().call_soon(entered.set)
        return await client.get(TREE, **kwargs)

    task = asyncio.create_task(get())
    await asyncio.wait_for(entered.wait(), 5)
    return task


@pytest.mark.parametrize("prior_fails", [False, True])
def test_mutation_waiters_share_successor_after_original_disconnect(monkeypatch, prior_fails):
    async def run():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event(), asyncio.Event()]
        release = [threading.Event(), threading.Event()]
        data = {"revision": "before"}
        calls = []

        def build(*args):
            index = len(calls)
            snapshot = dict(data)
            calls.append(snapshot)
            loop.call_soon_threadsafe(started[index].set)
            assert release[index].wait(10), "test did not release scan"
            if index == 0 and prior_fails:
                raise RuntimeError("old scan failed")
            return snapshot

        monkeypatch.setattr(profiles, "_build_profiles_projects_tree", build)
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = []
            try:
                original = await _request(client)
                tasks.append(original)
                await asyncio.wait_for(started[0].wait(), 5)
                original.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await original
                data["revision"] = "after"
                fresh = [await _request(client, params={"after_mutation": True}) for _ in range(8)]
                tasks.extend(fresh)
                # Canceling a freshness waiter must not cancel the shared prior scan.
                fresh[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await fresh[0]
                assert len(calls) == 1
                assert not any(task.done() for task in fresh[1:])
                assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
                release[0].set()
                await asyncio.wait_for(started[1].wait(), 5)
                assert not any(task.done() for task in fresh[1:])
                assert len(calls) == 2
                release[1].set()
                responses = await asyncio.gather(*fresh[1:])
                assert all(response.json() == data for response in responses)
                assert len(calls) == 2
            finally:
                for event in release:
                    event.set()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("variant", ["preview_limit", "session_limit", "home", "root", "native-root", "app",
                                     "cancel-leader", "cancel-all", "failure"])
def test_tree_flights_isolate_inputs_and_survive_disconnects(monkeypatch, tmp_path, variant):
    from pathlib import Path
    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("HERMES_HOME", "" if variant == "native-root" else str(tmp_path / "root"))

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls = []
        fail = variant == "failure"

        def build(preview_limit, session_limit):
            value = {"home": str(hermes_constants.get_hermes_home()),
                     "root": str(hermes_constants.get_default_hermes_root()),
                     "preview": preview_limit, "limit": session_limit}
            calls.append(value)
            loop.call_soon_threadsafe(started.set)
            assert release.wait(10), "test did not release tree build"
            if fail:
                raise RuntimeError("scan failed")
            return value

        monkeypatch.setattr(profiles, "_build_profiles_projects_tree", build)
        app = _app()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        other = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client, \
                httpx.AsyncClient(transport=other, base_url="http://test") as other_client:
            requests = []
            try:
                requests.append(await _request(client))
                await asyncio.wait_for(started.wait(), 5)
                requests.append(await _request(client))
                if variant.startswith("cancel"):
                    cancelled = requests[:1] if variant == "cancel-leader" else requests[:]
                    for task in cancelled:
                        task.cancel()
                    for task in cancelled:
                        with pytest.raises(asyncio.CancelledError):
                            await task
                params = {variant: 7} if variant in ("preview_limit", "session_limit") else {}
                with profiles._hermes_home_scope(tmp_path / "other") if variant == "home" else \
                        profiles._hermes_home_scope(hermes_constants.get_hermes_home()):
                    if variant == "root":
                        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-root"))
                    elif variant == "native-root":
                        monkeypatch.setattr(Path, "home", lambda: tmp_path / "other-native")
                        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "other-native"))
                    requests.append(await _request(other_client if variant == "app" else client,
                                                   params=params))
                assert (await asyncio.wait_for(client.get("/probe"), 2)).status_code == 200
            finally:
                release.set()
                responses = await asyncio.gather(*requests, return_exceptions=True)
            isolated = variant in ("preview_limit", "session_limit", "home", "root", "native-root", "app")
            assert len(calls) == (2 if isolated else 1)
            completed = [r for r in responses if isinstance(r, httpx.Response)]
            assert all(r.status_code == (500 if fail else 200) for r in completed)
            if variant in ("root", "native-root"):
                assert completed[0].json()["home"] == completed[-1].json()["home"]
                assert completed[0].json()["root"] != completed[-1].json()["root"]
            elif variant == "home":
                assert completed[0].json()["root"] == completed[-1].json()["root"]
                assert completed[0].json()["home"] != completed[-1].json()["home"]
            elif isolated and variant != "app":
                assert completed[0].json() != completed[-1].json()
            elif not fail:
                assert all(r.json() == completed[0].json() for r in completed)
            fail = False
            assert (await client.get(TREE)).status_code == 200
            assert len(calls) == (3 if isolated else 2)

    with profiles._hermes_home_scope(tmp_path / "fixed-home"):
        asyncio.run(run())
