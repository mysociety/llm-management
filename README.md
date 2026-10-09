# llm-management

A FastAPI app and CLI tool for managing [Exoscale](https://www.exoscale.com/) dedicated LLM inference deployments and prepared GPU VM servers.

See [Preparing Exoscale model servers](EXOSCALE_TEMPLATES.md) for packaging a
container and model into a VM template and configuring deployments to use it.

This acts as an intermediary between services/analysis processes and Exoscale, providing consistent endpoints, scale-to-zero, and specific functions with post-validation to centralise more complex LLM calls (e.g. extracting structure from FOI requests.)

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
[model.sar_logistic_v1]
repo = "mySociety/logistic-sar-detector-v1"
revision = "ade6333a06995cbde835648d758d8b092900bc52"

[[local.deployment]]
slug = "sar_logistic_v1_cpu"
loader = "sar_logistic_v1"
model_ref = "sar_logistic_v1"
artifact = "model.json"
```

Supported loaders are `modernbert_classifier`, `modernbert_head`,
`topic_tokenizer`, `sar_logistic_v1`, `sar_deberta_v1`, and `presidio`.
Each has a validated schema: classifiers and the topic tokenizer require
`max_tokens`; Presidio requires `spacy_model` and `score_threshold`.
Resource names come from their configured slugs. `[foi]`, `[sar]`, and
`[sanitization]` select those slugs for the application pipelines;
`[foi.topic]` defines the completion budget.

The catalog validates unique names, group membership, pinned local revisions,
and shared checkpoint references before registering cold resource factories.
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
| `/deployments/{slug}/ensure` | POST | Create or resume a deployment so it is running |
| `/deployments/{slug}/scale-to-zero` | POST | Scale a deployment to zero replicas (pause without destroying) |
| `/deployments/{slug}/v1/{path}` | POST | Proxy requests to the underlying Exoscale deployment, injecting auth |
| `/v1/systemone` | POST | Proxy System One decision requests to the selected Exoscale deployment |
| `/agents/sar_detection` | POST | Local personal-records moderation; failures retain an unclear flag |
| `/agents/immigration_detection/clef` | POST | Single Clef choice question returning `IMM` or `FOI` |
| `/agents/capital_city` | POST | Native structured-output example — returns a country's capital city |
| `/agents/foi_structure` | POST | QuestionSlice extraction followed by fine-tuned Granite regimes/topics; `backend=cpu` or `exoscale` selects extraction |
| `/agents/foi_structure/extract` | POST | QuestionSlice extraction only |
| `/local-models` | GET | Local CPU resource readiness and idle timers in this worker |
| `/local-models/{name}/ensure` | POST | Warm a local CPU resource and reset its idle timer |
| `/agents/foi_response_analysis` | POST | Response extraction v4 against upstream request questions; returns 501 until the response model is connected |
| `/agents/immigration_detection` | POST | Validated plain-text example — classifies a request as immigration-related (`IMM`) or FOI (`FOI`) |

### Clef / System One

The `clef` entry in `conf/deployments.toml` uses `backend = "exoscale_compute"` and
references the `clef_flash` recipe in `conf/exoscale_templates.toml`. The recipe
selects the prepared template, model revision and VM size. The checked-in recipe
points to the named template verified on 8 October 2026. See
[the template guide](EXOSCALE_TEMPLATES.md) for preparing another release.

The manager creates one GPU VM, connects through SSH, and waits for matching
model health. It reconnects to an existing VM after restart. Idle timeout and
shutdown delete the Compute VM and its access resources; managed deployments
continue scaling to zero. Persist `COMPUTE_STATE_DIR` (default `.state/compute`)
so manifests and private SSH keys survive restarts. One manager process/worker owns each
Compute deployment.

The server exposes `/v1/systemone?deployment=clef` and the native Pydantic AI
immigration agent at `/agents/immigration_detection/clef?deployment=clef`.
Model settings come from the template recipe. The serving container includes
Clef's decision head; Exoscale's managed vLLM gateway is not involved.

Call `/v1/systemone?deployment=clef` with the upstream server's JSON request
shape. Bodies and upstream query parameters are forwarded unchanged; the local
`deployment` selector is removed. Client credentials are removed before forwarding; managed deployment keys are
injected when present. The upstream status and body are returned to the caller. The
existing `/deployments/clef/v1/systemone` proxy can also forward these requests.

```bash
curl 'http://localhost:8000/v1/systemone?deployment=clef' \
  -H 'Authorization: Bearer <server-token>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"Cloudflare/clef-flash","state":"Please update me on my visa application.","questions":{"classification":{"type":"choice","instructions":"Classify this request.","criteria":{"IMM":"Immigration matters","FOI":"Other information requests"}}}}'
```

The minimal immigration endpoint uses a Pydantic AI `Agent` with a typed
`ClassificationResponse` output. Its native
[`SystemOneModel`](https://pydantic.dev/docs/ai/models/system-one/) translates
the classification field into a choice question and validates the returned
answer and probabilities. The provider uses the deployment's resolved API connection:


```bash
curl 'http://localhost:8000/agents/immigration_detection/clef?deployment=clef' \
  -H 'Authorization: Bearer <server-token>' \
  -H 'Content-Type: application/json' \
  -d '{"request":"Please update me on my visa application."}'
# {"classification":"IMM"}
```

Both endpoints use normal server authentication and `ensure_running`, which
updates the shared idle timer. Unknown deployment slugs return 404; transport
failures return 503 and timeouts return 504. The immigration endpoint returns
502 for upstream errors or malformed classifications. The original
`/agents/immigration_detection?deployment=toast_llama` remains available.

### Prepared GPU VM templates

The `llm-management templates` CLI uses recipes from
[conf/exoscale_templates.toml](conf/exoscale_templates.toml) to prepare and resolve
named VM templates. Its serving image is published from
[mysociety/systemone-container](https://github.com/mysociety/systemone-container/).
See [the template guide](EXOSCALE_TEMPLATES.md) for creation, testing and cleanup.

Temporary VMs, snapshots, SSH keys and groups are deleted; the reusable template is retained.
Live checks on 8 October 2026 verified offline model loading, HTTP and native
Pydantic AI requests, tunnel recovery and cleanup. A fresh template VM completed
its challenges in 5m 33s. The [template guide](EXOSCALE_TEMPLATES.md#live-lifecycle-verification)
records the template ID and lifecycle results.

### Automatic idle scaling

The server tracks when each deployment last received traffic. Deployments idle for longer than 15 minutes are stopped: managed inference scales
to zero, while Compute VMs and their access resources are deleted. Active HTTP
requests prevent idle teardown. Shutdown stops deployments used during the
session. Prepared templates are retained.

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

## FOI sanitization and local CPU lifecycle

Both FOI pipelines run Presidio locally on CPU before model inference. The request
pipeline uses sanitized text for both CPU and Exoscale QuestionSlice and for Granite
classification, while API extraction results retain the original request text.
The response pipeline sanitizes request context, question text, additional text,
source bodies, and filenames together; question/source IDs and structural metadata
are preserved. Response inference remains a placeholder until its model is connected.

The initial policy replaces person names, email addresses, phone numbers, credit
card/IBAN identifiers, UK NHS/National Insurance numbers, and obvious numbered UK
street addresses. It preserves ordinary dates, organisations, and geographic names.
Addresses use a conservative custom pattern; detection is best effort and does not
cover every possible personal identifier or address format. Replacements such as
`<PERSON_1>` are consistent for matching detected values within one analysis. Existing
placeholders are preserved; replacement maps are discarded after each operation.
A sanitizer failure returns 503 and blocks inference.

`poetry install` installs Presidio and the pinned English spaCy model. Presidio loads
that installed model explicitly on CPU and never downloads a model during a request.
An alternative `spacy_model` in the local Presidio deployment must be installed before starting the service.

Model resources share the same model-family/version name across runtimes:
`question_slice_v2` identifies the remote GPU deployment, `question_slice_v2_cpu`
the full local CPU model, and `question_slice_v2_head_cpu` its local classification
head used with GPU embeddings. Use these names in warm-up calls and groups.

Local resources load on demand. Use `POST /local-models/presidio/ensure`,
`POST /local-models/question_slice_v2_cpu/ensure`,
`POST /local-models/question_slice_v2_head_cpu/ensure`, or
`POST /local-models/question_extractor_tokenizer/ensure` to warm them explicitly. These endpoints
use the same authentication as the rest of the API. The shared
`POST /deployments/{slug}/ensure` endpoint also accepts registered local resource
names; local success returns `action: "warmed"` and `replicas: null`. The local
endpoint is an alias returning resource status. `GET /local-models` reports resource
controllers constructed in the current worker. Local resource names can also be
included alongside remote deployment slugs in deployment groups.

Idle cleanup drops model/tokenizer references after `CPU_IDLE_TIMEOUT_MINUTES`
(default 15), and the next request reloads them from the persistent model cache.
Active work is protected even when its HTTP caller disconnects. This is in-process
unloading: Python/native allocators may retain memory. Multiple API workers each own
their resources and idle timers, so warm-up reaches the worker handling that call.

External FOI adapters require `Sanitized[T]` and check its policy version at runtime.
Only sanitizers construct these wrappers; they store immutable JSON snapshots and
return fresh typed views. Chat payloads are sanitized after construction as well.
Future GPU adapters should require this contract and unwrap only at the transport
boundary. Raw proxy and other agent endpoints retain their existing contracts.

## Fine-tuned QuestionSlice extraction (experimental)

`POST /agents/foi_structure/extract?backend=cpu` runs the fine-tuned ModernBERT
question extractor. The body is `{"request": "..."}`. It returns reconstructed
questions, source-unit predictions/probabilities, additional/ignored text and an
`extraction_status` of `questions_found`, `no_questions_found` or `uncertain`.
If there are no predicted question starts anywhere but at least one continuation,
the first continuation is treated as a start during reconstruction. Later
continuations join that question using the existing grouping rules. The response
records `promoted_continuation_index` (otherwise null). Raw `unit_predictions`,
and probabilities remain unchanged for audit; the
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

`POST /agents/foi_response_analysis` accepts a `request` containing the complete
output of either request endpoint, or just its `questions`, `extraction_status`
and `additional_text`. It also requires a nonempty `sources` list describing
response emails/attachments. Each source has a unique `id`, `kind` (`email` or
`attachment`), `role` (`current`, `prior`, `quoted`), nullable `sender`
(`authority`, `requester`, `other`, `unknown`), nullable `filename`,
`availability` (`visible`, `unavailable`, `unknown`) and nullable `text`.
Visible sources require text; other sources require null text. Sender and
filename keys are required even when null.

```json
{
  "request": {
    "questions": [{"question_id": "q1", "text": "Please provide the report."}],
    "extraction_status": "questions_found",
    "additional_text": []
  },
  "request_text": "Please provide the report.",
  "sources": [{
    "id": "email1",
    "kind": "email",
    "role": "current",
    "sender": "authority",
    "filename": null,
    "availability": "visible",
    "text": "We do not hold this report."
  }]
}
```

The output contract follows the supplied response extraction v4 example:
`outcomes`, `events`, optional nullable `process_states`, and `process_references`.
Question/narrower scopes must reference upstream question IDs; supplied content
must reference provided source IDs with compatible visibility. Whole-request and
unresolved scopes remain available even when no questions were found.
Invalid inputs return 422; invalid model output returns 502. The response model
is still in training: the placeholder `foi-response-analysis-placeholder` returns
501 without provisioning a deployment or generating analysis. Replace `predict`
in `foi/response_analysis.py` when the checkpoint and serving contract are ready.

Contract differences from the request step:

- The training input needs only three request fields. Diagnostics (`ignored_text`,
  `unit_predictions`, extraction provenance and promotion index) and classification
  metadata are accepted in full upstream results but excluded from model input.
- Original request text lives inside the full upstream result, whereas v4 places
  it at the top level. The adapter copies it automatically; an explicit conflicting
  `request_text` is rejected. Extraction-only results require callers to supply
  original text separately if they want that optional context.
- `classification.questions` contains regime/topic metadata, not question text.
  Response analysis uses the top-level reconstructed `questions` array. The v4
  model does not consume topics or regimes or return the request-step diagnostics.
- The existing request schemas do not enforce unique IDs or status/list consistency.
  Response analysis enforces both v4 invariants without changing the request endpoints.
- A response is represented as identified sources, rather than a single raw string.
  Ingestion diagnostics must be converted explicitly to v4 availability values.

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
Granite substitution is allowed. The deployment and local tokenizer must reference the same `[model.<name>]` entry;
update its `repo` and `revision` together to keep them aligned. Model merging and publication are handled in
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

This uses the `question_slice_v2` deployment in `conf/deployments.toml`, with the
existing ensure/resume/idle-scaling lifecycle. Exoscale's gateway does not expose
vLLM's native `/classify`, so this backend serves the **fine-tuned encoder** through
`/v1/embeddings` with mean pooling and normalisation disabled, then applies the
**original fine-tuned classification head** locally. This is the same trained model
split across devices, not an unadapted embedding model or a replacement classifier.
Prefix caching must be disabled for this encoder on the tested vLLM 0.29.0 runtime.

GPU probabilities may differ slightly due to float16 execution. Keep the imported
encoder and local head from the same checkpoint release. The local head uses the shared checkpoint’s pinned `revision`;
remote responses use `extraction_revision: null` because the Exoscale importer does not verify
the encoder SHA. Loading the head currently downloads the full safetensors artifact
into the cache, but retains only its small head tensors for inference.

There is no silent backend fallback. Set `QUESTION_SLICE_GPU_ENABLED=false` to
disable remote extraction without provisioning anything.

Optional environment settings:

| Setting | Default | Purpose |
|---|---|---|
| `DEPLOYMENT_CONFIG` | `conf/deployments.toml` | Deployment catalog path |
| `CPU_INFERENCE_THREADS` | `1` | Process-wide PyTorch intra-op thread count |
| `CPU_IDLE_TIMEOUT_MINUTES` | `15` | Unload unused local CPU resources; checked every minute |
| `CLASSIFIER_BATCH_SIZE` | `8` | Maximum semantic units per inference call |
| `CLASSIFIER_MAX_UNITS` | `256` | Maximum semantic units accepted per request |
| `CLASSIFIER_CACHE_DIR` | Hugging Face default | Persistent tokenizer/weight cache location |
| `QUESTION_SLICE_GPU_ENABLED` | `true` | Allow the tested Exoscale encoder + local-head backend |

Model identities and limits are validated by `DeploymentCatalog`. The maximum topic question count is derived from the output
budget. Change limits only to values supported by the trained checkpoint.

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
| `foi/schemas.py` | Labels and request-stage input/output structures |
| `foi/response_schemas.py` | Strict response extraction v4 contracts and upstream request adapter |
| `foi/response_analysis.py` | Response model placeholder and output reference validation |
| `foi/question_slice.py` | Segmentation, contextual windows, probability validation and question reconstruction |
| `foi/backends.py` | Cached classifier/head factories and CPU versus Exoscale execution |
| `foi/question_extractor.py` | Tokenizer loading, training-compatible prompts and constrained topic output |
| `deployments.py` | Typed catalog, checkpoint references, pipeline bindings and limits |
| `inference.py` / `errors.py` | Reusable classifier execution and runtime errors, independent of FOI code |
| `settings.py` | Credentials and host environment settings; model and deployment config is in `conf/deployments.toml` |
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
`test_inference.py` covers reusable runtime behavior, `test_question_extractor.py` covers prompts
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

Named groups in `conf/deployments.toml` let batch clients warm remote deployments and registered local CPU resources concurrently:

```toml
[[deployment_group]]
slug = "foi_pipeline"
deployments = [
    "presidio", "question_slice_v2_head_cpu", "question_extractor_tokenizer",
    "question_slice_v2", "foi_topic_v2",
]

[[deployment_group]]
slug = "foi_pipeline_cpu"
deployments = ["presidio", "question_slice_v2_cpu", "question_extractor_tokenizer", "foi_topic_v2"]
```

Call `POST /deployment-groups/foi_pipeline/ensure` with the usual authentication before a batch using GPU extraction. This warms Presidio, the local classification head and Granite tokenizer while creating or resuming both GPU deployments, even when automatic startup on inference requests is disabled. For CPU extraction, use `POST /deployment-groups/foi_pipeline_cpu/ensure`, which warms the full CPU classifier and the remote topic model. Existing per-deployment locks prevent overlapping group and inference requests from starting the same deployment twice within one API process.

A known group returns HTTP 200 with `slug`, overall `success`, and an ordered `deployments` list. Each member has `slug`, `success`, `replicas` (null for local resources or on failure) and `error` (null on success). Inspect the success flags: a partial or complete startup failure is reported in the body. Unknown groups return 404. Successful members are not rolled back when another fails; retrying the group reuses loaded local resources and running deployments. Idle scaling and shutdown cleanup follow each resource's lifecycle, and warm-up does not keep a group running indefinitely.

Groups must be nonempty, have unique names, and reference local or Exoscale deployment slugs declared in the catalog without repeated members. Local and remote names must not collide. Catalog validation checks these references before registering cold factories, without importing resource owners or loading models. Adding a local deployment with a supported loader makes it available to individual and group warm-up without changing the HTTP handlers.

### SAR moderation on local CPU

`POST /agents/sar_detection` accepts `{"request": "the complete original correspondence"}`
and returns `is_sar`, `status` (`complete` or `unclear`), `reason`,
`logistic_score` and `deberta_score`. It flags requests for personal records,
including personal immigration correspondence and mixed personal/public requests;
it is not a general personal-information detector. This endpoint uses the usual
API authentication and does not start an Exoscale deployment or run the FOI pipeline.

The strict logistic model receives raw text. Every logistic positive proceeds to
DeBERTa using the vendored predictor's placeholder removal, NFKC, casefold and
whitespace normalization. Only a successful negative can clear the flag. Loading,
inference, invalid-score, overlength and busy failures return HTTP 200 with
`is_sar: true`, `status: "unclear"` and a reason. Invalid HTTP request bodies still
receive normal validation errors. Treat unclear results as requiring moderator review.
Scores describe model outputs, not established real-world probabilities.

Models are lazy local resources named `sar_logistic_v1_cpu` and
`sar_deberta_v1_cpu`. Before a batch, call
`POST /deployment-groups/sar_pipeline_cpu/ensure`, or use individual
`POST /local-models/{name}/ensure` calls. Inspect the group's member success flags.
Warm-up and loading are per API worker, and resources follow the normal CPU idle
unloading policy. Allow roughly 1.28 GB for the reported SAR process footprint,
plus other resources loaded in the same worker. DeBERTa uses CPU float32, the shared
`CPU_INFERENCE_THREADS` setting (default 1), and one in-flight inference per worker;
concurrent DeBERTa calls return an unclear busy flag. Logistic negatives can continue
while DeBERTa is busy. Blocking inference runs off the event loop.

Private artifact access uses Pydantic Settings: `HF_TOKEN` takes precedence over
legacy `HUGGINGFACE_TOKEN`. `CLASSIFIER_CACHE_DIR` controls the Hugging Face cache;
mount it persistently in containers to avoid repeat downloads. Tokens and model
weights are not included in the repository. Artifact identities are pinned in
`conf/deployments.toml`; trained cutoff checks remain in the detector implementation:

| Model | Commit | Inclusive positive cutoff |
|---|---|---|
| `mySociety/logistic-sar-detector-v1` | `ade6333a06995cbde835648d758d8b092900bc52` | JSON cutoff `0.0018603434604847564` |
| `mySociety/deberta-sar-detecter-v1` | `199883a0387af896f34f7bf0978472268a9474b9` | Softmax index 1, `0.5` |

DeBERTa rejects inputs above 512 tokens including special tokens without truncation.
The upstream artifacts record Unicode database **16.0.0**. This service intentionally
relaxes the upstream runtime-version check and uses the host Python's Unicode tables
(Python 3.11 uses Unicode 14.0.0). Placeholder removal, NFKC, casefold, whitespace
collapse and logistic word-ngram tokenization are unchanged. Python/runtime and ML
dependency versions remain unchanged. This is a documented compatibility adaptation:
characters whose normalization, casing or word membership changed between Unicode
versions may produce different features or prepared text. Representative correspondence
comparisons do not establish parity for every possible Unicode input. A comparison
of 20 synthetic samples (including accented/decomposed names, several scripts,
whitespace, placeholders and emoji) between Python 3.11 / Unicode 14 and Python
3.14 / Unicode 16 found identical prepared text, active logistic features, scores
and logistic decisions. This checks compatibility on those examples, not accuracy
on real correspondence.

The shipped pair still needs validation on fresh real correspondence; historical procedure metrics do not validate
this deployment. Vendored inference code is MIT; cached model artifacts are
Apache-2.0 with upstream DeBERTa MIT notices retained in their repositories.
