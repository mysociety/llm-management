"""HTTP tests use an independent application service and cache."""

import pytest


@pytest.fixture(autouse=True)
def isolated_application(monkeypatch, request):
    if request.node.get_closest_marker("external") or not hasattr(
        request.module, "server"
    ):
        return
    from llm_management import server

    monkeypatch.setattr(server, "app", server.create_app())
