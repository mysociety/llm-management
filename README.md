# llm-management

A FastAPI app and CLI for managing Exoscale inference deployments and prepared GPU VMs.
Deployment configuration lives in `conf/deployments.toml`; prepared template recipes
live in `conf/exoscale_templates.toml`.

## Setup

Install with `poetry install`, then start `llm-management serve` (or use Docker Compose).
Set Exoscale credentials and the server role through environment variables or `.env`.
`AUTH_TOKENS` is a JSON mapping of client names to bearer tokens. Health and API docs
remain public; an empty token mapping leaves API requests unauthenticated.

## Deployment operations

CLI commands create, destroy, pause, resume, connect, list, show logs and test deployments.
`llm-management templates` prepares reusable VM images. See
[template preparation](EXOSCALE_TEMPLATES.md), [deployment management](docs/deployments.md)
and [System One](docs/systemone.md).

HTTP management supports status, ensure, replica overrides, scale-to-zero, cooldown
and named-group warm-up. Managed deployments scale to zero when idle; prepared Compute
VMs and their access resources are deleted. Inspect group member results even when
HTTP returns 200. One application service owns locks, backend controllers and activity.

## Validation

Run `poetry run pytest -m 'not external' -q`, `poetry run ruff check .` and
`poetry run ruff format --check .`. External tests contact or create cloud resources.
