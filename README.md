# llm-management

A FastAPI app and CLI tool for managing [Exoscale](https://www.exoscale.com/) dedicated LLM inference deployments and prepared GPU VM servers.

See [Preparing Exoscale model servers](EXOSCALE_TEMPLATES.md) for packaging a
container and model into a VM template and configuring deployments to use it.

This acts as an intermediary between services/analysis processes and Exoscale, providing consistent endpoints, scale-to-zero, and specific functions with post-validation to centralise more complex LLM calls (e.g. extracting structure from FOI requests.)

Development conventions are documented in [Coding style](docs/coding-style.md).

## Configuration

### Environment variables

Set the following via environment variables or a `.env` file in the working directory:

- `EXOSCALE_API_KEY` — Your Exoscale API key
- `EXOSCALE_API_SECRET` — Your Exoscale API secret
- `EXOSCALE_SERVER_ROLE` — Role suffix appended to deployment names on Exoscale (e.g. `test`, `production`). Defaults to `test`
- `AUTH_TOKENS` — Optional JSON mapping of client names to bearer tokens. Defaults to an empty mapping (no auth)
- `COMPUTE_STATE_DIR` — Persistent directory for Compute manifests and private SSH keys; defaults to `.state/compute`

### Deployment config

Deployments are defined in `conf/deployments.toml`. Each `[[exoscale.deployment]]` block describes a remote deployment:

```toml
[[exoscale.deployment]]
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

Local resources use `[[local.deployment]]` with a `slug` and a typed `loader`.
Checkpoint identities live once in `[model.<name>]`, referenced by `model_ref`:

```toml
[model.sar_logistic]
repo = "mySociety/sar-detect-logistic-2026-10-09-1"
revision = "ade6333a06995cbde835648d758d8b092900bc52"

[[local.deployment]]
slug = "sar_logistic_cpu"
loader = "sar_logistic"
model_ref = "sar_logistic"
artifact = "model.json"
```

Supported loaders are `modernbert_classifier`, `modernbert_head`,
`topic_tokenizer`, `sar_logistic`, `sar_deberta`, `auto_logistic`,
`auto_deberta`, and `presidio`.
Each has a validated schema: classifiers and the topic tokenizer require
`max_tokens`; Presidio requires `spacy_model` and `score_threshold`.
Resource names come from their configured slugs. `[foi]`, `[sar]`, `[automatic]`, and
`[sanitization]` select those slugs for the application pipelines;
`[foi.topic]` defines the completion budget.

Catalog loading validates unique names, group membership, pinned local revisions,
and shared checkpoint references. Application assembly then registers cold resource factories.
Loading config or registering factories does not download or load model weights.
Remote Exoscale commands and `--all` operate on `exoscale.deployment` only.
Template preparation recipes remain in `conf/exoscale_templates.toml`.

Set `DEPLOYMENT_CONFIG` to use another catalog file; restart the process after
changing configuration. This replaces `conf/exoscale.toml` and its top-level
`[[deployment]]` entries. Model identities, pipeline bindings, token limits,
and Presidio options now come from the catalog; the previous `FOI_TOPIC_*`,
`QUESTION_SLICE_MODEL`, `QUESTION_SLICE_REVISION`, `QUESTION_SLICE_INPUT_LIMIT`,
`QUESTION_SLICE_DEPLOYMENT`, and `PRESIDIO_*` environment overrides have been removed.
Credentials, cache paths, host settings, and `QUESTION_SLICE_GPU_ENABLED` remain
environment settings.

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
| `logs SLUG [--tail N]` | Show managed deployment logs or the Compute boot service's journal |
| `llm-test [basic\|instruct] SLUG` | Test a deployment by asking the LLM for the capital of France |
| `clear-models ZONE [ID] [--all]` | Remove model(s) from a zone (fails if in use by a deployment) |

Most commands accept a deployment `SLUG` (matching a slug in `conf/deployments.toml`) or `--all` to apply to every configured deployment.

## FastAPI proxy server

The package includes a FastAPI server that acts as an intermediary between services and Exoscale deployments. Start it with:

```
llm-management serve [--host 0.0.0.0] [--port 8000] [--reload]
```

The server provides interactive API docs at the root URL (`/`).

### Server authentication

If `AUTH_TOKENS` is an empty JSON object, the FastAPI server does not require authentication.

If the mapping contains any items, API requests must include a token from the mapping:

```bash
AUTH_TOKENS='{"analysis-service":"token-one","admin-tool":"token-two"}'
```

Give each client its own clearly named token so it can be revoked independently.
Generate with `openssl rand -hex 32`

```http
Authorization: Bearer <token>
```

Requests with a missing or incorrect token are rejected with `401 Not authenticated`.
The health endpoint and API documentation (`/`, `/openapi.json`, and `/redoc`) remain
available without authentication.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Unauthenticated process health check |
| `/deployments` | GET | Overview of all configured deployments with idle timers |
| `/deployments/{slug}/status` | GET | Check whether a deployment exists and its replica count |
| `/deployments/{slug}/ensure` | POST | Create/resume a deployment; optional `replicas` query parameter also resizes running deployments |
| `/deployments/{slug}/cooldown` | POST | Shorten the remaining idle countdown to at most one minute |
| `/deployments/{slug}/scale-to-zero` | POST | Scale a deployment to zero replicas (pause without destroying) |
| `/deployments/{slug}/v1/{path}` | POST | Proxy requests to the underlying Exoscale deployment, injecting auth |
| `/v1/systemone` | POST | Proxy System One decision requests to the selected Exoscale deployment |
| `/agents/sar_detection` | POST | Local personal-records moderation; failures retain an unclear flag |
| `/agents/automatic_response_detection` | POST | Classify one email body as automatic or authored; supports fast logistic-only mode |
| `/agents/immigration_detection/clef` | POST | Single Clef choice question returning `IMM` or `FOI` |
| `/agents/capital_city` | POST | Native structured-output example — returns a country's capital city |
| `/agents/foi_structure` | POST | QuestionSlice extraction followed by fine-tuned Granite regimes/topics; `backend=cpu` or `exoscale` selects extraction |
| `/agents/foi_structure/extract` | POST | QuestionSlice extraction only |
| `/local-models` | GET | Local CPU resource readiness and idle timers in this worker |
| `/local-models/{name}/ensure` | POST | Warm a local CPU resource and reset its idle timer |
| `/agents/foi_response_analysis` | POST | Response extraction v4 against upstream request questions; returns 501 until the response model is connected |
| `/agents/immigration_detection` | POST | Validated plain-text example — classifies a request as immigration-related (`IMM`) or FOI (`FOI`) |

## Common operations

Start the API with `llm-management serve`. Warm a named deployment or local resource
with `POST /deployments/{slug}/ensure`, or prepare a complete batch pipeline with
`POST /deployment-groups/{slug}/ensure`. Group responses report each member's success;
inspect the body even when HTTP returns 200.

An optional `replicas` query parameter on deployment ensure resizes managed
inference deployments. Compute supports one replica. `POST /deployments/{slug}/cooldown`
shortens the remaining idle timer to at most a minute; new traffic resets it.
Managed deployments scale to zero when idle; prepared Compute VMs and their access
resources are deleted. Local model weights unload independently when idle.

The Clef routes use prepared VM templates and System One transport. FOI routes use
QuestionSlice extraction and remote Granite topic classification. SAR and automatic
response detection run locally. FOI response-analysis contracts are implemented;
model inference is pending and currently returns 501.

## Run with Docker

The repository includes a `Dockerfile` and `docker-compose.yml` for running the proxy server in a container.

### 1. Create a `.env` file

The compose setup loads environment variables from `.env`:

```bash
EXOSCALE_API_KEY=your_key
EXOSCALE_API_SECRET=your_secret
SERVER_ROLE=test
AUTH_TOKENS={}
```

### 2. Build and start

You can use the helper scripts in `script/`:

```bash
script/build
script/server
```

- `script/build` runs `docker compose build`.
- `script/server` checks `IN_DOCKER`:
    - when `IN_DOCKER` is set (inside the container), it runs `poetry run llm-management serve`
    - when `IN_DOCKER` is not set (on the host), it runs `docker compose up`

Equivalent raw Docker Compose commands:

```bash
docker compose build
docker compose up
```

Then open `http://localhost:8080/` for the API docs.

To stop the app:

```bash
docker compose down
```

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

External tests provision the deployments selected by each test module. The full
FOI pipeline can keep both the ModernBERT encoder and Granite running together
when testing GPU extraction. Test-client shutdown scales touched deployments to
zero; verify cleanup after interruptions or provisioning failures.

### Markers

| Marker | Description |
|---|---|
| `external` | Test creates or connects to a real Exoscale deployment. These tests require valid `EXOSCALE_API_KEY` / `EXOSCALE_API_SECRET` credentials, will start GPU instances, and may take several minutes. |

Tests marked `external` cover the proxy, Toast classification, the full fine-tuned FOI pipeline, and ModernBERT CPU/Exoscale parity. They use FastAPI's `TestClient` to run the server in-process while provisioning real infrastructure. App shutdown scales deployments touched by the tests to zero; check deployment state after interrupted runs or provisioning failures. Unmarked tests run locally without real model inference.

To test the full pipeline with both extraction backends:

```bash
script/test --all -m external tests/test_foi_pipeline_external.py -v
```

To run only the paid ModernBERT parity checks:

```bash
script/test --all -m external tests/test_question_slice_external.py -v
```

Three checks compare CPU/GPU labels and reconstructed questions, including text
without questions and a continuation recovered by the local heuristic. A fourth
check verifies the exact recovered single question. No OLMo deployment is used by
these extraction tests. These smoke tests do not establish production accuracy.

## Guides and source navigation

| Guide | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Ownership, dependency direction, reading paths and adding features |
| [Deployment management](docs/deployments.md) | Lifecycle, replica overrides, cooldown and group warm-up |
| [FOI processing](docs/foi.md) | Extraction, topics, pending response analysis, provenance and batch examples |
| [Sanitization and resources](docs/sanitization.md) | PII policy, attestations, leases and idle unloading |
| [Correspondence classifiers](docs/classifiers.md) | SAR and automatic-response decisions, fast modes and failure policies |
| [Clef and System One](docs/systemone.md) | Proxy and typed immigration classification |
| [VM template preparation](EXOSCALE_TEMPLATES.md) | Preparing and verifying reusable model-server templates |
| [Shared GPU serving proposal](docs/proposals/shared-gpu-serving.md) | Proposed Compute extraction and LoRA serving; not implemented |

Start reading application code at `bootstrap.py` and `server.py`. HTTP contracts
live under `api/`; deployment lifecycle and cloud operations live under
`deployments/`; reusable execution and resource guards live under `inference/`.
FOI, SAR and automatic classification each keep their own decisions and schemas.
