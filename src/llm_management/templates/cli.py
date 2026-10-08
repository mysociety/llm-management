"""CLI for configured template releases."""

import json
import functools
from pathlib import Path

import typer

from ..models import LLMManagementError, get_client
from .config import TEMPLATE_CONFIG_PATH, TemplateConfig, TemplateRecipe
from .lifecycle import build, cleanup

app = typer.Typer(help="Build and resolve named Exoscale VM templates.")


def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMManagementError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapper


@app.callback()
def configure(
    ctx: typer.Context, config: Path = typer.Option(TEMPLATE_CONFIG_PATH, "--config")
) -> None:
    ctx.obj = config


def recipe_for(ctx: typer.Context, slug: str) -> TemplateRecipe:
    return TemplateConfig.load(ctx.obj).get(slug)


@app.command("create")
@handle_errors
def create(
    ctx: typer.Context,
    slug: str,
    runs: int = typer.Option(2, min=1),
    ssh_cidr: str | None = typer.Option(None),
) -> None:
    """Build a new named template and test fresh VMs; incurs cloud charges."""
    recipe = recipe_for(ctx, slug)
    if ssh_cidr:
        recipe = recipe.model_copy(update={"ssh_cidr": ssh_cidr})
    build(recipe, runs)


@app.command("test")
@handle_errors
def test(
    ctx: typer.Context,
    slug: str,
    runs: int = typer.Option(2, min=1),
    ssh_cidr: str | None = typer.Option(None),
) -> None:
    """Boot/test an existing template by configured name, then delete test VMs."""
    recipe = recipe_for(ctx, slug)
    if ssh_cidr:
        recipe = recipe.model_copy(update={"ssh_cidr": ssh_cidr})
    build(recipe, runs, test_only=True)


@app.command("resolve")
@handle_errors
def resolve(ctx: typer.Context, slug: str) -> None:
    """Print the UUID for the configured private template name and zone."""
    recipe = recipe_for(ctx, slug)
    typer.echo(recipe.resolve(get_client(recipe.zone))["id"])


@app.command("list")
@handle_errors
def list_templates(ctx: typer.Context) -> None:
    """List recipes and whether their named templates are registered."""
    config = TemplateConfig.load(ctx.obj)
    clients = {zone: get_client(zone) for zone in {r.zone for r in config.template}}
    for recipe in config.template:
        template = recipe.find(clients[recipe.zone])
        typer.echo(
            f"{recipe.slug}\t{recipe.zone}\t{recipe.name}\t{template['id'] if template else 'not created'}"
        )


@app.command("cleanup")
@handle_errors
def recover(
    state_file: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Recover temporary resources after interruption; retain registered templates."""
    state = json.loads(state_file.read_text())
    cleanup(get_client(state["zone"]), state, state_file)
