"""Public documentation separates inference, lifecycle, and example contracts."""

from llm_management import server


def test_documented_route_groups():
    paths = server.app.openapi()["paths"]
    for path in (
        "/endpoint/foi/sar_detection",
        "/endpoint/foi/automatic_response_detection",
        "/endpoint/foi/extract_questions",
        "/endpoint/foi/extract_questions/extract",
        "/endpoint/foi/response_status",
        "/endpoint/foi/find_email",
    ):
        assert paths[path]["post"]["tags"] == ["FOI endpoints"]
    assert paths["/endpoint/misc/capital_city"]["post"]["tags"] == ["Misc endpoints"]
    for path in (
        "/mock/endpoint/misc/capital_city",
        "/mock/endpoint/foi/extract_questions",
        "/mock/endpoint/foi/extract_questions/extract",
        "/mock/endpoint/foi/response_status",
        "/mock/endpoint/foi/find_email",
    ):
        assert paths[path]["post"]["tags"] == ["Mocks"]
    for action in ("warm-up", "cool-down", "scale-to-zero"):
        assert paths[f"/deployments/{{slug}}/{action}"]["post"]["tags"] == [
            "Deployments"
        ]
    assert paths["/local-deployments"]["get"]["tags"] == ["Deployments"]
    assert paths["/deployments/{slug}/v1/{path}"]["post"]["tags"] == ["Inference proxy"]


def test_examples_are_hidden_and_obsolete_routes_removed():
    paths = server.app.openapi()["paths"]
    registered = {route.path for route in server.app.routes}
    for path in (
        "/endpoint/foi/immigration_detection",
        "/endpoint/foi/immigration_detection/clef",
    ):
        assert path in registered
        assert path not in paths
    assert "/v1/systemone" not in registered
    assert not any(path.startswith("/deployment-group/") for path in registered)
    assert not any("/agents/" in path or "/local-models" in path for path in paths)
    assert "/local-deployments/{name}/ensure" not in registered
    assert "/local-models/{name}/ensure" not in registered


def test_documented_operations_declare_bearer_auth():
    schema = server.app.openapi()
    scheme = schema["components"]["securitySchemes"]["BearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"
    for path in schema["paths"].values():
        for method, operation in path.items():
            if method in {"get", "post", "put", "patch", "delete", "options", "head"}:
                assert operation["security"] == [{"BearerAuth": []}]
                assert not any(
                    p["name"].lower() == "authorization"
                    for p in operation.get("parameters", [])
                )


def test_email_finder_mock_uses_the_live_input_and_response_contract() -> None:
    paths = server.app.openapi()["paths"]
    live = paths["/endpoint/foi/find_email"]["post"]
    mock = paths["/mock/endpoint/foi/find_email"]["post"]
    assert mock["requestBody"] == live["requestBody"]
    assert mock["responses"]["200"] == live["responses"]["200"]


def test_warmup_action_schema_lists_supported_outcomes() -> None:
    schemas = server.app.openapi()["components"]["schemas"]
    assert schemas["WarmUpResponse"]["properties"]["action"] == {
        "$ref": "#/components/schemas/WarmUpAction"
    }
    assert schemas["WarmUpAction"]["type"] == "string"
    assert set(schemas["WarmUpAction"]["enum"]) == {
        "warmed",
        "scaled",
        "already_running",
        "resumed",
        "created",
    }
