# llm-management

A CLI tool for managing [Exoscale](https://www.exoscale.com/) dedicated LLM inference deployments.

## Configuration

### Environment variables

Set the following via environment variables or a `.env` file in the working directory:

- `EXOSCALE_API_KEY` — Your Exoscale API key
- `EXOSCALE_API_SECRET` — Your Exoscale API secret
- `EXOSCALE_SERVER_ROLE` — Role suffix appended to deployment names on Exoscale (e.g. `test`, `production`). Defaults to `test`

### Deployment config

Deployments are defined in `conf/exoscale.toml`. Each `[[deployment]]` block describes a model to deploy:

```toml
[[deployment]]
slug = "olmo3"
model = "allenai/Olmo-3-7B-Instruct"
gpu_type = "gpua5000"
gpu_count = 1
replicas = 1
zone = "at-vie-2"
inference_engine_params = [
    "--enable-prefix-caching",
    "--enable-auto-tool-choice",
    "--tool-call-parser=olmo3"
]
```

| Field | Description |
|---|---|
| `slug` | Unique name for the deployment (used locally and in CLI/API) |
| `model` | Model identifier (uploaded to the zone if not already present) |
| `gpu_type` | GPU type (e.g. `gpua5000`) |
| `gpu_count` | Number of GPUs |
| `replicas` | Number of replicas |
| `zone` | Exoscale zone (e.g. `at-vie-2`) |
| `inference_engine_params` | Optional vLLM engine parameters |

### Deployment naming

The `slug` is the local identifier used in CLI commands, API routes, and the cache. On Exoscale, deployments are created with the name `{slug}_{server_role}` (e.g. `olmo3_7b_test` or `olmo3_7b_production`). This allows test and production instances to share the same config file and Exoscale account without interfering with each other — a test instance will never find or modify a production deployment, and vice versa.

## CLI usage

```
llm-management [COMMAND] [OPTIONS]
```

### Commands

| Command | Description |
|---|---|
| `create [SLUG] [--all] [--refresh-model]` | Create deployment(s), uploading the model to the zone if needed. `--refresh-model` deletes and re-uploads the model first |
| `destroy [SLUG] [--all]` | Delete deployment(s) |
| `pause [SLUG] [--all]` | Scale deployment(s) to zero replicas |
| `resume SLUG` | Resume a paused deployment to its configured replica count |
| `create-or-resume SLUG` | Create the deployment if it doesn't exist, or resume it if paused |
| `connect SLUG` | Show the connection URL and API key for a deployment |
| `list` | List all deployments across configured zones |
| `list-models ZONE` | List all models in a zone |
| `logs SLUG [--tail N]` | Show recent log output for a deployment |
| `llm-test [basic\|instruct] SLUG` | Test a deployment by asking the LLM for the capital of France |
| `clear-models ZONE [ID] [--all]` | Remove model(s) from a zone (fails if in use by a deployment) |

Most commands accept a deployment `SLUG` (matching a slug in `conf/exoscale.toml`) or `--all` to apply to every configured deployment.

## FastAPI proxy server

The package includes a FastAPI server that acts as an intermediary between your services and Exoscale deployments. Start it with:

```
llm-management serve [--host 0.0.0.0] [--port 8000] [--reload]
```

The server provides interactive API docs at the root URL (`/`).

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/deployments` | GET | Overview of all configured deployments with idle timers |
| `/deployments/{slug}/status` | GET | Check whether a deployment exists and its replica count |
| `/deployments/{slug}/ensure` | POST | Create or resume a deployment so it is running |
| `/deployments/{slug}/scale-to-zero` | POST | Scale a deployment to zero replicas (pause without destroying) |
| `/deployments/{slug}/v1/{path}` | POST | Proxy requests to the underlying Exoscale deployment, injecting auth |
| `/agents/capital_city` | POST | Example agent — returns the capital city of a given country |
| `/agents/immigration_detection` | POST | Example agent — classifies a request as immigration-related (`IMM`) or FOI (`FOI`) |

### Automatic idle scaling

The server tracks when each deployment last received traffic. Deployments that have been idle for longer than 15 minutes are automatically scaled to zero. On shutdown, all deployments that received traffic during the session are also scaled to zero.

### Agent endpoints

Agent endpoints wrap [pydantic-ai](https://docs.pydantic.dev/ai/) agents with built-in system prompts and structured output. Each agent runs against a specific deployment (configurable via query parameter). New agents can be added under `src/llm_management/agents/` and registered in `server.py`.

## Testing

Tests use [pytest](https://docs.pytest.org/) and live under `tests/`.

### Running tests

```bash
# Run only fast, local tests (no external services needed)
script/test

# Run all tests including those that create Exoscale deployments
script/test --all
```

Additional pytest arguments are passed through, e.g. `script/test -v` or `script/test --all -k toast`.

### Markers

| Marker | Description |
|---|---|
| `external` | Test creates or connects to a real Exoscale deployment. These tests require valid `EXOSCALE_API_KEY` / `EXOSCALE_API_SECRET` credentials, will start GPU instances, and may take several minutes. |

Tests marked `external` (in `test_proxy.py` and `test_toast.py`) use FastAPI's `TestClient` to run the server in-process but call `/deployments/{slug}/ensure`, which provisions real infrastructure. The remaining tests (`test_meta.py`, `test_llm_management.py`) are purely local and need no credentials or network access.