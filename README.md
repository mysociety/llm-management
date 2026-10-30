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
- `AUTH_TOKENS` — JSON mapping of client names to bearer tokens. Required to start the HTTP server unless local testing is explicitly enabled.
- `ALLOW_UNAUTHENTICATED` — Defaults to `false`. Set to `true` only for local testing with an empty `AUTH_TOKENS` mapping.
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
`auto_deberta`, `response_tokenizer`, and `presidio`.
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

The HTTP server refuses to start with an empty `AUTH_TOKENS` mapping unless
`ALLOW_UNAUTHENTICATED=true` is explicitly set. CLI cloud-management commands
do not require HTTP authentication settings.

For testing without tokens, explicitly set these values in `.env`:

```dotenv
AUTH_TOKENS={}
ALLOW_UNAUTHENTICATED=true
```

`ALLOW_UNAUTHENTICATED` defaults to `false`. This setting controls authentication
only; the existing bind addresses and Docker port mappings are unchanged.

If the mapping contains any items, API requests must include a token from the mapping,
even when the testing flag is enabled:

```bash
AUTH_TOKENS='{"analysis-service":"token-one","admin-tool":"token-two"}'
```

Give each client its own clearly named token so it can be revoked independently.
All tokens intentionally have the same inference and management permissions.
Generate with `openssl rand -hex 32`

```http
Authorization: Bearer <token>
```

Requests with a missing or incorrect token are rejected with `401 Not authenticated`.
The health endpoint and API documentation (`/`, `/openapi.json`, and `/redoc`) remain
available without authentication.

To authenticate **Try it out** requests in the interactive docs at `/`, click
**Authorize**, paste a token value from `AUTH_TOKENS` (without the `Bearer `
prefix), and confirm. Swagger adds the Authorization header automatically.
Use **Logout** in that dialog to clear the token; authorization is not configured
to persist across page reloads.

### Endpoints

The interactive documentation groups routes into FOI endpoints, Misc endpoints,
Deployments, Inference proxy, and Mocks.

Clients using the former `/agents/...`, `/mock/agents/...`, `/deployment-groups/...`,
or `/local-models` URLs must migrate to the paths below. Local warm-up now uses
`/deployments/{slug}/warm-up`; its success response has `action: "warmed"` and
`replicas: null`. Read local readiness from `/local-deployments`. The action URLs use `/warm-up`
and `/cool-down`; the former `/ensure` and `/cooldown` routes are removed.

#### FOI endpoints

| Endpoint | Method | Description | Relevant deployment group |
|---|---|---|---|
| `/endpoint/foi/extract_questions` | POST | Given an FOI request, returns questions with FOI/EIR/SAR status, short descriptions and local predefined category matches | `group_foi_pipeline_cpu` (default CPU extraction), or `group_foi_pipeline` (`backend=exoscale`) |
| `/endpoint/foi/extract_questions/extract` | POST | Just the question structure, primarily for internal debugging; omits status and topic classification | Same groups as above; these warm the full pipeline |
| `/endpoint/foi/response_status` | POST | Given a structured set of questions and a structured response document, identifies whether questions were answered or withheld | `group_foi_response_pipeline` |
| `/endpoint/foi/sar_detection` | POST | Given an FOI request text, indicates whether it is likely to be a SAR, IMM, or other personal information request; `fast=true` uses only logistic inference rather than cascading to the larger model | `group_sar_pipeline_cpu` |
| `/endpoint/foi/automatic_response_detection` | POST | Given a response to an FOI request, indicates whether it is likely to be an automatic response; `fast=true` uses only logistic inference rather than cascading to the larger model | `group_automatic_pipeline_cpu` |

Warm a group with `POST /deployments/{group_slug}/warm-up`.

#### Deployment management and inference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Unauthenticated process health check (hidden from OpenAPI) |
| `/deployments` | GET | Overview of all configured deployments with idle timers |
| `/deployments/{slug}/status` | GET | Check whether a deployment exists and its replica count |
| `/deployments/{slug}/warm-up` | POST | Warm a local resource, remote deployment, or `group_` group; optional `replicas` applies only to individual remote deployments |
| `/deployments/{slug}/cool-down` | POST | Shorten idle timers for one resource or every group member to at most one minute |
| `/deployments/{slug}/scale-to-zero` | POST | Stop/unload one resource or every group member, protecting active work |
| `/local-deployments` | GET | Local CPU resource readiness and idle timers in this worker |
| `/deployments/{slug}/v1/{path}` | POST | Proxy requests, including System One (`path=systemone`), injecting upstream auth |

#### Misc endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/endpoint/misc/capital_city` | POST | Structured-output example using model inference; returns a country's capital city |

#### Mocks and examples

| Endpoint | Method | Description |
|---|---|---|
| `/mock/endpoint/foi/extract_questions` | POST | Example extraction and classification without inference |
| `/mock/endpoint/foi/extract_questions/extract` | POST | Example extraction without inference |
| `/mock/endpoint/foi/response_status` | POST | Example response analysis without inference |
| `/mock/endpoint/foi/find_email` | POST | Fixed fictional FOI email finding with zero external usage |
| `/mock/endpoint/misc/capital_city` | POST | Always returns `city: "Fake Paris"` and echoes the submitted country, without inference |

The immigration examples (`/endpoint/foi/immigration_detection` and its `/clef`
variant) remain callable but are hidden from OpenAPI. The standalone `/v1/systemone`
route has been removed; use `/deployments/{slug}/v1/systemone` instead.

## Common operations

Start the API with `llm-management serve`. Warm a named deployment, local resource,
or group with `POST /deployments/{slug}/warm-up`. Group slugs start with `group_`,
for example `/deployments/group_foi_response_pipeline/warm-up`. Configuration
validation reserves that prefix for groups. The former `/deployment-group/...`
routes have been removed. Group responses report each member's success; inspect
the body even when HTTP returns 200.

An optional `replicas` query parameter on deployment warm-up resizes managed
inference deployments. Compute supports one replica. `POST /deployments/{slug}/cool-down`
shortens the remaining idle timer to at most a minute; new traffic resets it.
Managed deployments scale to zero when idle; prepared Compute VMs and their access
resources are deleted. Local model weights unload independently when idle.
Groups support the same three actions under `/deployments/{slug}/...`.
Group calls return per-member results, including busy or failed members; inspect
`success` even when HTTP returns 200. These actions also handle local resources.

The Clef routes use prepared VM templates and System One transport. FOI routes use
QuestionSlice extraction and remote Granite topic classification. SAR and automatic
response detection run locally. FOI response analysis uses the configured Granite
status checkpoint with the exact published v4 prompt and schema. It is an integration
test candidate; validate results before using them for operational decisions.

## Run with Docker

The repository includes a `Dockerfile` and `docker-compose.yml` for running the proxy server in a container.

### 1. Create a `.env` file

The compose setup loads environment variables from `.env`:

```bash
EXOSCALE_API_KEY=your_key
EXOSCALE_API_SECRET=your_secret
SERVER_ROLE=test
AUTH_TOKENS={"production-client":"replace-with-a-generated-token"}
ALLOW_UNAUTHENTICATED=false
```

For testing without tokens, use `AUTH_TOKENS={}` and explicitly set
`ALLOW_UNAUTHENTICATED=true` in `.env`. Compose publishes `8080:8080` as before.

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

Plain `pytest` skips live tests, including when `-m external` is selected. Direct
pytest invocations must add `--run-external`; `script/test --all` supplies this
flag. Live runs require the configured `SERVER_ROLE` to be `test` or `test_<name>`;
production and empty roles are rejected before application fixtures start.
Each live session replaces that role with a fresh `test_<random ID>` and uses
`COMPUTE_STATE_DIR/pytest/<role>` for recovery state. The role and directory are
printed during setup (use `-s` to show them for successful tests). Existing
deployments are not reused across test sessions. Keep the recovery directory if
a run is interrupted, and use its role and path for manual cleanup.

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
| [Resource registration](docs/resource-registration.md) | Resource families, type versus instance, and how to add a model approach |
| [Deployment management](docs/deployments.md) | Lifecycle, replica overrides, cooldown and group warm-up |
| [FOI processing](docs/foi.md) | Extraction, topics, pending response analysis, provenance and batch examples |
| [Sanitization and resources](docs/sanitization.md) | PII policy, attestations, leases and idle unloading |
| [Correspondence classifiers](docs/classifiers.md) | SAR and automatic-response decisions, fast modes and failure policies |
| [Clef and System One](docs/systemone.md) | Proxy and typed immigration classification |
| [VM template preparation](EXOSCALE_TEMPLATES.md) | Preparing and verifying reusable model-server templates |
| [Shared GPU serving proposal](docs/proposals/shared-gpu-serving.md) | Proposed Compute extraction and LoRA serving; not implemented |

Start with `bootstrap.py` for runtime assembly and `http_app.py` for HTTP construction.
`server.py` is the ASGI entry point that binds them together. HTTP contracts live under `api/`; deployment lifecycle and cloud operations live under
`deployments/`; reusable execution and resource guards live under `inference/`.
FOI, SAR and automatic classification each keep their own decisions and schemas.

### Request and inference limits

Requests are limited before JSON parsing or model work. These per-worker settings
can be adjusted through environment variables:

| Setting | Default | Behaviour |
| --- | --- | --- |
| `HTTP_MAX_BODY_BYTES` | 1,048,576 | Oversized requests return 413, including chunked uploads. |
| `HTTP_MAX_CONCURRENT_REQUESTS` | 64 | Excess concurrent requests return 429 with `Retry-After`. |
| `HTTP_BODY_TIMEOUT_SECONDS` | 30 | Incomplete uploads return 408. |
| `PROXY_MAX_RESPONSE_BYTES` | 8,388,608 | Oversized upstream proxy responses return 502. |
| `LOGISTIC_MAX_PENDING` | 64 | Bounds retained logistic jobs per classifier, including queued or caller-cancelled jobs; overload returns 429. |

Automatic-response text is limited to 100,000 characters. Response-status input
allows up to 64 sources, 256 questions and 256 additional-text items; individual
source text is limited to 100,000 characters, and combined model-input text to
200,000 characters. Model token budgets still apply.

Configured authentication token values must be nonblank ASCII strings. An empty
`AUTH_TOKENS` mapping requires the explicit `ALLOW_UNAUTHENTICATED=true` local
testing opt-in; otherwise HTTP startup fails.

The proxy rejects dot segments, backslashes and residual percent encoding in its
path before starting a deployment. Query parameters remain supported.

Docker builds use a deny-by-default `.dockerignore` that admits application
source, package metadata and the two deployment configuration files. Supply
credentials at runtime; local environment files, state, SSH keys and Git history
are excluded from the build context.

The public-authority research endpoint, `POST /endpoint/foi/find_email`, finds
published FOI submission addresses with source evidence using external APIs.
See [API deployment configuration, credentials and data policy](docs/foi.md#find-an-authoritys-foi-email).
