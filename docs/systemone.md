# Clef and System One

The `clef` entry in `conf/deployments.toml` uses `backend = "exoscale_compute"` and
references the `clef_flash` recipe in `conf/exoscale_templates.toml`. The recipe
selects the prepared template, model revision and VM size. The checked-in recipe
points to the named template verified on 8 October 2026. See
[the template guide](../EXOSCALE_TEMPLATES.md) for preparing another release.

The manager creates one GPU VM, connects through SSH, and waits for matching
model health. It reconnects to an existing VM after restart. Idle timeout and
shutdown delete the Compute VM and its access resources; managed deployments
continue scaling to zero. Persist `COMPUTE_STATE_DIR` (default `.state/compute`)
so manifests and private SSH keys survive restarts. One manager process/worker owns each
Compute deployment.

The server exposes `/v1/systemone?deployment=clef` and the native Pydantic AI
immigration agent at `/endpoint/foi/immigration_detection/clef?deployment=clef`.
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

The immigration examples remain callable but are hidden from OpenAPI.
The minimal immigration endpoint uses a Pydantic AI `Agent` with a typed
`ClassificationResponse` output. Its native
[`SystemOneModel`](https://pydantic.dev/docs/ai/models/system-one/) translates
the classification field into a choice question and validates the returned
answer and probabilities. The provider uses the deployment's resolved API connection:


```bash
curl 'http://localhost:8000/endpoint/foi/immigration_detection/clef?deployment=clef' \
  -H 'Authorization: Bearer <server-token>' \
  -H 'Content-Type: application/json' \
  -d '{"body":"Please update me on my visa application."}'
# {"classification":"IMM"}
```

Both endpoints use normal server authentication and `ensure_running`, which
updates the shared idle timer. Unknown deployment slugs return 404; transport
failures return 503 and timeouts return 504. The immigration endpoint returns
502 for upstream errors or malformed classifications. The original
`/endpoint/foi/immigration_detection?deployment=toast_llama` remains available.
