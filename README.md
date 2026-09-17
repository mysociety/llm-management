# llm-management

A FastAPI app and CLI tool for managing [Exoscale](https://www.exoscale.com/) dedicated LLM inference deployments.

This acts as an intermediary between services/analysis processes and Exoscale, providing consistent endpoints, scale-to-zero, and specific functions with post-validation to centralise more complex LLM calls (e.g. extracting structure from FOI requests.)

## Configuration

### Environment variables

Set the following via environment variables or a `.env` file in the working directory:

- `EXOSCALE_API_KEY` — Your Exoscale API key
- `EXOSCALE_API_SECRET` — Your Exoscale API secret
- `EXOSCALE_SERVER_ROLE` — Role suffix appended to deployment names on Exoscale (e.g. `test`, `production`). Defaults to `test`
- `AUTH_TOKENS` — Optional JSON mapping of client names to bearer tokens. Defaults to an empty mapping (no auth)

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
| `/deployments/{slug}/ensure` | POST | Create or resume a deployment so it is running |
| `/deployments/{slug}/scale-to-zero` | POST | Scale a deployment to zero replicas (pause without destroying) |
| `/deployments/{slug}/v1/{path}` | POST | Proxy requests to the underlying Exoscale deployment, injecting auth |
| `/agents/capital_city` | POST | Native structured-output example — returns a country's capital city |
| `/agents/foi_structure` | POST | QuestionSlice extraction followed by fine-tuned Granite regimes/topics; `backend=cpu` or `exoscale` selects extraction |
| `/agents/foi_structure/extract` | POST | QuestionSlice extraction only |
| `/agents/immigration_detection` | POST | Validated plain-text example — classifies a request as immigration-related (`IMM`) or FOI (`FOI`) |

### Automatic idle scaling

The server tracks when each deployment last received traffic. Deployments that have been idle for longer than 15 minutes are automatically scaled to zero. On shutdown, all deployments that received traffic during the session are also scaled to zero.

### Agent endpoints

Agent endpoints wrap [pydantic-ai](https://docs.pydantic.dev/ai/) agents with built-in system prompts and structured output. Each agent runs against a specific deployment (configurable via query parameter). New agents can be added under `src/llm_management/agents/` and registered in `server.py`.

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

## Fine-tuned QuestionSlice extraction (experimental)

`POST /agents/foi_structure/extract?backend=cpu` runs the fine-tuned ModernBERT
question extractor. The body is `{"request": "..."}`. It returns reconstructed
questions, source-unit predictions/probabilities, additional/ignored text and an
`extraction_status` of `questions_found`, `no_questions_found` or `uncertain`.
If there are no predicted question starts anywhere but at least one continuation,
the first continuation is treated as a start during reconstruction. Later
continuations join that question using the existing grouping rules. The response
records `promoted_continuation_index` (otherwise null). Raw `unit_predictions`,
probabilities and `orphan_continuation_indices` remain unchanged for audit; the
final questions, residual text and `extraction_status` reflect the recovery.

If a start already exists, orphan continuations remain `uncertain`. If neither
starts nor continuations exist, the result remains `no_questions_found`. No OLMo
escalation is performed. This dataset-specific heuristic can merge separate asks;
its boundary accuracy still requires evaluation on representative correspondence.

This extraction endpoint does not classify topics/regimes. Use
`POST /agents/foi_structure?backend=cpu` (or `backend=exoscale`) for the complete
pipeline. The same request body is used. Granite always runs remotely on the
`foi_topic_v2` Exoscale deployment after extraction; no Granite call is made when
there are no extracted questions. Remaining `uncertain` status is preserved even
when Granite classifies the questions that were successfully extracted.

The full response includes all extraction fields plus `request_text`,
`classification` and model provenance. `classification.questions` contains ordered
`question_id`, `regime` (`FOI`, `EIR`, `SAR`) and `topic`; `classification.request_topics`
contains unique request-level topics. Question IDs match the source-grounded
`questions` array exactly. `classification` and `classification_model` are null when no
classification is needed. The old summary, five keywords and `ir_type` schema,
metadata endpoint and legacy extraction flow have been removed.

Example full-pipeline request:

```bash
curl -X POST 'http://localhost:8080/agents/foi_structure?backend=cpu' \
  -H 'Content-Type: application/json' \
  -d '{"request":"Please provide the latest air pollution monitoring report."}'
```

Granite serves `mySociety/granite-tiny-foi-topic-grounded-v2-merged`, containing the
`grounded-v2` adapter merged into Granite 4.0 1B. The merged artifact was inspected
at revision `c2ba6b86e43977bcb71bc90f12dc0cad42ac7e79`; its tokenizer/chat template
is pinned locally to that revision. Exoscale imports weights by repository name,
so keep that import and local tokenizer in sync when updating the model. No base
Granite substitution is allowed. Model merging and publication are handled in
the fine-tuning project; this service does not require PEFT.

Granite uses JSON-schema constrained generation and validates IDs, counts, regimes,
and unique topics after inference. Inputs over 2,048 chat-template tokens are
rejected before provisioning Granite. Output allowance is `max(256, 64 + 192*n)`,
capped at 2,048 tokens (at most ten questions); exceeding either input or output
budget returns 422 without truncation or automatic chunking. Invalid/incomplete
upstream output returns 502; unavailable deployment/tokenizer returns 503.
This budget remains a heuristic, and smoke tests do not establish accuracy.

Both extraction backends are included in the default Docker image. For a non-Docker
installation, install the inference dependencies in the Python environment running
the API:

```bash
poetry install
```

With Docker, use the normal build and start commands:

```bash
docker compose build
docker compose up
```

Choose CPU or Exoscale per request using the `backend` query parameter; no build-time
mode selection is needed. Both use the same CPU Torch package: CPU mode runs the full
model locally, while Exoscale mode runs only the small classification head locally.
Exoscale-only use does not load the full CPU encoder.
Poetry locks the inference dependencies together with the API dependencies and
uses the explicit PyTorch CPU package source. The setup was tested on Linux x86_64
with Python 3.11. Mount `CLASSIFIER_CACHE_DIR` on
persistent storage to retain downloaded weights across container replacements.

Example (add the usual bearer header when `AUTH_TOKENS` is configured):

```bash
curl -X POST 'http://localhost:8080/agents/foi_structure/extract?backend=cpu' \
  -H 'Content-Type: application/json' \
  -d '{"request":"Please provide the annual expenditure report."}'
```

Use the same request with `backend=exoscale` for GPU-backed extraction:

```bash
curl -X POST 'http://localhost:8080/agents/foi_structure/extract?backend=exoscale' \
  -H 'Content-Type: application/json' \
  -d '{"request":"Please provide the annual expenditure report."}'
```

This uses the `question_slice_v2` deployment in `conf/exoscale.toml`, with the
existing ensure/resume/idle-scaling lifecycle. Exoscale's gateway does not expose
vLLM's native `/classify`, so this backend serves the **fine-tuned encoder** through
`/v1/embeddings` with mean pooling and normalisation disabled, then applies the
**original fine-tuned classification head** locally. This is the same trained model
split across devices, not an unadapted embedding model or a replacement classifier.
Prefix caching must be disabled for this encoder on the tested vLLM 0.29.0 runtime.

GPU probabilities may differ slightly due to float16 execution. Keep the imported
encoder and local head from the same checkpoint release. The local head uses `QUESTION_SLICE_REVISION`;
remote responses use `extraction_revision: null` because the Exoscale importer does not verify
the encoder SHA. Loading the head currently downloads the full safetensors artifact
into the cache, but retains only its small head tensors for inference.

There is no silent backend fallback. Set `QUESTION_SLICE_GPU_ENABLED=false` to
disable remote extraction without provisioning anything.

Optional environment settings:

| Setting | Default | Purpose |
|---|---|---|
| `FOI_TOPIC_DEPLOYMENT` | `foi_topic_v2` | Merged fine-tuned Granite deployment for the full FOI pipeline |
| `CPU_INFERENCE_THREADS` | `1` | Process-wide PyTorch intra-op thread count |
| `CPU_INFERENCE_PRELOAD` | `false` | Load CPU model during application startup instead of the first CPU request |
| `CLASSIFIER_BATCH_SIZE` | `8` | Maximum semantic units per inference call |
| `CLASSIFIER_MAX_UNITS` | `256` | Maximum semantic units accepted per request |
| `CLASSIFIER_CACHE_DIR` | Hugging Face default | Persistent tokenizer/weight cache location |
| `QUESTION_SLICE_MODEL` | `mySociety/modernbert-question-slice-v2` | Classifier checkpoint |
| `QUESTION_SLICE_REVISION` | `eb0436d4be96f113f35b5f891f9cc876a0f0bd6b` | Pinned local tokenizer/model revision |
| `QUESTION_SLICE_DEPLOYMENT` | `question_slice_v2` | Remote deployment slug |
| `QUESTION_SLICE_GPU_ENABLED` | `true` | Allow the tested Exoscale encoder + local-head backend |

A model instance is cached per API worker process. CPU inference runs outside the
async event loop with one in-flight request per model instance. Overload returns
429 with `Retry-After: 1`; batch clients should retry with backoff and limit parallel
requests. This is backpressure, not a durable job queue or cross-request batching.
Requests above 100,000 characters, the configured unit count, or 768 tokens in any
context window are rejected with 422. Inputs are never silently truncated.
Unavailable dependencies/models return 503 and malformed upstream outputs return
502. Avoid multiple API workers until deployment lifecycle ownership is coordinated;
the existing Exoscale cache and idle scaler are process-local.

## Code navigation

The FOI pipeline lives in `src/llm_management/foi/`. Start with
`pipeline.py`: `extract_questions()` performs extraction, and
`process_information_request()` adds Granite classification. Both are async Python
functions that accept request text, the extraction backend, and a `DeploymentAccess`
object supplying deployment lookup, ensure/resume and activity tracking. They can
be called from batch code without a FastAPI request or `TestClient`.

| Module | Responsibility |
|---|---|
| `foi/pipeline.py` | Stage ordering, skipped classification and domain errors |
| `foi/schemas.py` | Labels and input/output structures shared by both stages |
| `foi/question_slice.py` | Segmentation, contextual windows, probability validation and question reconstruction |
| `foi/backends.py` | Cached classifier/head factories and CPU versus Exoscale execution |
| `foi/granite.py` | Tokenizer loading, training-compatible prompts and constrained topic output |
| `foi/model_spec.py` | Checkpoint identities, revisions and model input/output limits |
| `inference.py` / `errors.py` | Reusable classifier execution and runtime errors, independent of FOI code |
| `settings.py` | Runtime settings/environment variables; deployment hardware is in `conf/exoscale.toml` |
| `server.py` | HTTP contracts, error translation and existing deployment lifecycle wiring |

`agents/` contains the actual Pydantic AI agent implementations. FOI extraction
and classification no longer live there. Keep schemas free of runtime imports;
model-specific factories belong in `foi/backends.py`, not the generic classifier.

Within reconstruction, `predictions_from_probabilities()` preserves raw labels,
`promote_initial_continuation_if_needed()` applies the narrow recovery rule, and
`build_extraction_result()` assembles raw diagnostics with the recovered questions.
The original orphan indices may therefore remain present when the final status is
`questions_found`; `promoted_continuation_index` explains the recovery.

Tests follow these boundaries: `test_question_slice.py` covers pure processing,
`test_inference.py` covers reusable runtime behavior, `test_granite.py` covers prompts
and transport validation, `test_foi_pipeline.py` calls the pipeline directly, and
`test_foi_routes.py` checks HTTP contracts. The existing external test modules
exercise real models through the unchanged API routes.

### Model identity and provenance

Responses identify extraction with `extraction_model`, `extraction_revision` and
`extraction_backend`. The `backend` query parameter still selects extraction.
`classification_model` identifies the merged checkpoint used for classification,
or is null when classification is skipped. No separate adapter is loaded at runtime.

The merged Granite checkpoint was built from `ibm-granite/granite-4.0-1b`
(revision `6a7381ba1f54d684ff508d991aeb7dc580157103`) and
`mySociety/granite-tiny-foi-topic-grounded-v2`
(revision `200b756850c137c255f2e6c5c24474edd894d010`). These describe training
provenance; they are not independently served models.

## Deployment groups

Named groups in `conf/exoscale.toml` let batch clients prepare several deployments concurrently:

```toml
[[deployment_group]]
slug = "foi_pipeline"
deployments = ["question_slice_v2", "foi_topic_v2"]
```

Call `POST /deployment-groups/foi_pipeline/ensure` with the usual authentication before a batch using GPU extraction. This explicitly creates or resumes both deployments, even when automatic startup on inference requests is disabled. Existing per-deployment locks prevent overlapping group and inference requests from starting the same deployment twice within one API process.

A known group returns HTTP 200 with `slug`, overall `success`, and an ordered `deployments` list. Each member has `slug`, `success`, `replicas` (null on failure) and `error` (null on success). Inspect the success flags: a partial or complete startup failure is reported in the body. Unknown groups return 404. Successful members are not rolled back when another fails; retrying the group reuses running deployments. Idle scaling and shutdown cleanup remain per deployment, and warm-up does not keep a group running indefinitely.

Groups must be nonempty, have unique names, and reference existing deployment slugs without repeated members.