"""
CLI for managing Exoscale dedicated inference deployments.

https://exoscale.github.io/python-exoscale/index.html
"""

from __future__ import annotations

import json
from typing import Literal, Optional

import rich
import typer

from .models import ExoscaleConfig, LLMManagementError, get_client

app = typer.Typer(help="Manage Exoscale dedicated inference deployments.")


def handle_errors(func):
    """Decorator to catch LLMManagementError and exit cleanly."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMManagementError as e:
            rich.print(f"Error: {e}")
            raise typer.Exit(1)

    return wrapper


@app.command()
@handle_errors
def create(
    slug: Optional[str] = typer.Argument(
        None, help="Deployment slug from exoscale.toml"
    ),
    all_: bool = typer.Option(False, "--all", help="Apply to all deployments"),
    refresh_model: bool = typer.Option(
        False, "--refresh-model", help="Delete and re-upload the model before deploying"
    ),
):
    """Create deployment(s). Ensures the model exists in the zone first."""
    config = ExoscaleConfig.load()
    for cfg in config.resolve(slug, all_):
        cfg.create_deployment(refresh_model=refresh_model)


@app.command()
@handle_errors
def destroy(
    slug: Optional[str] = typer.Argument(
        None, help="Deployment slug from exoscale.toml"
    ),
    all_: bool = typer.Option(False, "--all", help="Apply to all deployments"),
):
    """Delete deployment(s)."""
    config = ExoscaleConfig.load()
    for cfg in config.resolve(slug, all_):
        cfg.delete_deployment()


@app.command("pause")
@handle_errors
def pause(
    slug: Optional[str] = typer.Argument(
        None, help="Deployment slug from exoscale.toml"
    ),
    all_: bool = typer.Option(False, "--all", help="Apply to all deployments"),
):
    """Scale deployment(s) to zero replicas."""
    config = ExoscaleConfig.load()
    for cfg in config.resolve(slug, all_):
        cfg.scale_to_zero()


@app.command()
@handle_errors
def resume(
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
):
    """Resume a deployment from zero replicas."""
    config = ExoscaleConfig.load()
    config.get(slug).resume_from_zero()


@app.command("create-or-resume")
@handle_errors
def create_or_resume(
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
):
    """Create or resume a deployment."""
    config = ExoscaleConfig.load()
    config.get(slug).create_or_resume()


@app.command()
@handle_errors
def connect(
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
):
    """Show connection URL and API key for a deployment."""
    config = ExoscaleConfig.load()
    info = config.get(slug).connection_info()
    json_info = json.dumps(info, indent=2)
    rich.print(f"{json_info}")


@app.command("list")
@handle_errors
def list_deployments():
    """List all currently deployed Exoscale models across configured zones."""
    config = ExoscaleConfig.load()
    config.list_deployments()


@app.command("list-models")
@handle_errors
def list_models(
    zone: str = typer.Argument(..., help="Exoscale zone (e.g. at-vie-2)"),
):
    """List all models in a zone."""
    client = get_client(zone)
    models = client.list_models().get("models", [])
    if not models:
        rich.print(f"No models in zone {zone}.")
        return
    for m in models:
        rich.print(f"  {m.get('name', 'unnamed')}  ({m.get('id', 'no-id')})")


@app.command("clear-models")
@handle_errors
def clear_models(
    zone: str = typer.Argument(..., help="Exoscale zone (e.g. at-vie-2)"),
    identifier: Optional[str] = typer.Argument(None, help="Model ID or name to delete"),
    all_: bool = typer.Option(False, "--all", help="Delete all models in the zone"),
):
    """Remove one model by ID or name, or all models from a zone."""
    if not identifier and not all_:
        rich.print("Error: Provide a model ID/name or use --all.")
        raise typer.Exit(1)
    client = get_client(zone)
    deployments = client.list_deployments().get("deployments", [])

    def model_in_use(model_id: str, model_name: str) -> list[str]:
        """Return names of deployments using this model."""
        using = []
        for d in deployments:
            dep_model = d.get("model", {})
            if dep_model.get("id") == model_id or dep_model.get("name") == model_name:
                using.append(d.get("name", "unnamed"))
        return using

    def find_model_by_name(name: str) -> dict | None:
        models = client.list_models().get("models", [])
        for m in models:
            if m.get("name") == name:
                return m
        return None

    if all_:
        models = client.list_models().get("models", [])
        if not models:
            rich.print(f"No models in zone {zone}.")
            return
        for m in models:
            using = model_in_use(m["id"], m["name"])
            if using:
                rich.print(
                    f"Error: Model {m['name']} ({m['id']}) is used by deployment(s): "
                    f"{', '.join(using)}. Delete those deployments first."
                )
                raise typer.Exit(1)
        for m in models:
            rich.print(f"Deleting model {m['name']} ({m['id']})...")
            op = client.delete_model(id=m["id"])
            client.wait(op["id"])
            rich.print("Deleted.")
    else:
        if not identifier:
            rich.print("Error: Provide a model ID/name or use --all.")
            raise typer.Exit(1)
        # Try to resolve identifier as a name first, then fall back to ID
        model = find_model_by_name(identifier)
        if model:
            model_id = model["id"]
            model_name = model["name"]
        else:
            model_id = identifier
            model_info = client.get_model(id=model_id)
            model_name = model_info.get("name", "")
        using = model_in_use(model_id, model_name)
        if using:
            rich.print(
                f"Error: Model {model_name or model_id} is used by deployment(s): "
                f"{', '.join(using)}. Delete those deployments first."
            )
            raise typer.Exit(1)
        rich.print(
            f"Deleting model {model_name or model_id} ({model_id}) from zone {zone}..."
        )
        op = client.delete_model(id=model_id)
        client.wait(op["id"])
        rich.print("Deleted.")


@app.command()
@handle_errors
def logs(
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
    tail: int = typer.Option(100, "--tail", "-n", help="Number of log lines to show"),
):
    """Show log tail for a deployment."""
    config = ExoscaleConfig.load()
    cfg = config.get(slug)
    client = cfg._client()
    deployment = cfg._find_deployment(client)
    if deployment is None:
        rich.print(f"Deployment {slug} not found.")
        raise typer.Exit(1)
    deploy_id = deployment["id"]
    state = deployment.get("state", "unknown")
    state_details = deployment.get("state-details", "")
    rich.print(f"Deployment {slug} (state: {state})")
    if state_details:
        rich.print(f"Details: {state_details}")
    rich.print("")
    log_data = client.get_deployment_logs(id=deploy_id, tail=tail)
    for entry in log_data.get("logs", []):
        time = entry.get("time", "")
        message = entry.get("message", "")
        rich.print(f"{time}  {message}")


@app.command("llm-test")
@handle_errors
def llm_test(
    mode: Literal["basic", "instruct"] = typer.Argument(
        ..., help="Test mode: 'basic' or 'instruct'"
    ),
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
):
    """Test a deployment by asking the LLM for the capital of France."""
    if mode not in ("basic", "instruct"):
        rich.print(f"Error: mode must be 'basic' or 'instruct', got '{mode}'.")
        raise typer.Exit(1)
    config = ExoscaleConfig.load()
    cfg = config.get(slug)
    rich.print(f"Testing deployment {slug} ({mode})...")
    if mode == "basic":
        cfg.test_basic_deployment()
    else:
        cfg.test_instruct_deployment()
    rich.print(f"[green]Test passed:[/green] deployment {slug} is working correctly.")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
):
    """Start the FastAPI proxy server."""
    import uvicorn

    uvicorn.run(
        "llm_management.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
