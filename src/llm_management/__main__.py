"""
CLI for managing Exoscale dedicated inference deployments.

https://exoscale.github.io/python-exoscale/index.html
"""

from __future__ import annotations

import json
from typing import Optional

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
):
    """Create deployment(s). Ensures the model exists in the zone first."""
    config = ExoscaleConfig.load()
    for cfg in config.resolve(slug, all_):
        cfg.create_deployment()


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


@app.command("clear-models")
@handle_errors
def clear_models(
    zone: str = typer.Argument(..., help="Exoscale zone (e.g. at-vie-2)"),
    id: Optional[str] = typer.Argument(None, help="Model ID to delete"),
    all_: bool = typer.Option(False, "--all", help="Delete all models in the zone"),
):
    """Remove one model by ID, or all models from a zone."""
    if not id and not all_:
        rich.print("Error: Provide a model ID or use --all.")
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
        # Look up the model to get its name for the in-use check
        model_info = client.get_model(id=id)
        using = model_in_use(id or "", model_info.get("name", ""))
        if using:
            rich.print(
                f"Error: Model {id} is used by deployment(s): "
                f"{', '.join(using)}. Delete those deployments first."
            )
            raise typer.Exit(1)
        rich.print(f"Deleting model {id} from zone {zone}...")
        op = client.delete_model(id=id)
        client.wait(op["id"])
        rich.print("Deleted.")


@app.command()
@handle_errors
def logs(
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of log lines to show"),
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
    slug: str = typer.Argument(..., help="Deployment slug from exoscale.toml"),
):
    """Test a deployment by asking the LLM for the capital of France."""
    config = ExoscaleConfig.load()
    cfg = config.get(slug)
    rich.print(f"Testing deployment {slug}...")
    cfg.test_deployment()
    rich.print(f"[green]Test passed:[/green] deployment {slug} is working correctly.")


if __name__ == "__main__":
    app()
