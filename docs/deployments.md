# Deployment management and warm-up

Internal model keys and resource slugs describe their role (`question_slice`,
`foi_topic`, `sar_deberta`) rather than a checkpoint version. Dated Hugging Face
repository names and pinned revisions identify the actual releases.

Managed cloud names are derived from the deployment slug and server role. Changing
a slug changes the cloud name addressed by the application; it does not rename an
existing Exoscale deployment.

### Prepared GPU VM templates

The `llm-management templates` CLI uses recipes from
[conf/exoscale_templates.toml](../conf/exoscale_templates.toml) to prepare and resolve
named VM templates. Its serving image is published from
[mysociety/systemone-container](https://github.com/mysociety/systemone-container/).
See [the template guide](../EXOSCALE_TEMPLATES.md) for creation, testing and cleanup.

Temporary VMs, snapshots, SSH keys and groups are deleted; the reusable template is retained.
Live checks on 8 October 2026 verified offline model loading, HTTP and native
Pydantic AI requests, tunnel recovery and cleanup. A fresh template VM completed
its challenges in 5m 33s. The [template guide](../EXOSCALE_TEMPLATES.md#live-lifecycle-verification)
records the template ID and lifecycle results.

To request a replica count, call `POST /deployments/{slug}/warm-up?replicas=3`.
The count must be a positive integer. Omitting it uses the configured count for
creation/resume and leaves running deployments at their current count. The
configuration is not changed. Resizing a running deployment returns action
`scaled`. Compute deployments support only `replicas=1`; local CPU resources
reject this parameter.

### Automatic idle scaling

The server tracks when each deployment last received traffic. Deployments idle for longer than 15 minutes are stopped: managed inference scales
to zero, while Compute VMs and their access resources are deleted. Active HTTP
requests prevent idle teardown. Shutdown stops deployments used during the
session. Prepared templates are retained.

Call `POST /deployments/{slug}/cool-down` after a batch to reduce the remaining
idle countdown to at most 60 seconds. A shorter countdown is preserved. The
response includes `seconds_until_scale_to_zero`; already stopped deployments
return `already_stopped` with a null countdown. Active requests return HTTP 409.
Subsequent traffic or warm-up calls reset the normal idle timer. The background
scaler checks every 60 seconds, so shutdown can occur up to one minute after
the shortened countdown expires.

## Deployment groups

Groups use the same lifecycle URLs as individual deployments. Every group slug
must start with `group_` followed by a nonempty name; individual local and remote
deployments cannot use that prefix. Catalog validation enforces this at load time.
For example, use `group_foi_response_pipeline` as `{slug}` below. The separate
`/deployment-group/...` routes are removed. Group warm-up rejects `replicas`
overrides with HTTP 422 before starting any members.

Groups support these authenticated actions for both local and remote members:

- `POST /deployments/{slug}/warm-up` loads local resources and starts remote deployments.
- `POST /deployments/{slug}/cool-down` shortens each member's idle countdown to at most 60 seconds. New activity or warm-up resets it.
- `POST /deployments/{slug}/scale-to-zero` unloads local resources, pauses managed inference, and deletes Compute VMs and their access resources immediately.

All actions run members concurrently and preserve configured order in the results.
Busy members fail without interruption; other members still proceed. A known group
returns HTTP 200 even on partial failure, so inspect overall and per-member `success`.
Unknown groups return 404. Cool-down and scale-to-zero results include `slug`,
`success`, `action`, `seconds_until_scale_to_zero`, and `error` per member. The
countdown is populated only for a running member that was cooled down; stopped
members return `action: "already_stopped"`. Successful stops return
`action: "scaled_to_zero"`, including local resource unloading.

The individual `/deployments/{slug}/cool-down` and `/scale-to-zero` actions also
accept local resources. Local expiry uses the same periodic idle sweep; local
cleanup does not guarantee immediate memory reclamation by the OS. Group actions
operate on shared resources, so active work is protected but an idle resource may
also belong to another group.

Named groups in `conf/deployments.toml` let batch clients warm remote deployments and registered local CPU resources concurrently:

```toml
[[deployment_group]]
slug = "group_foi_pipeline"
deployments = [
    "presidio", "question_slice_head_cpu", "question_extractor_tokenizer",
    "question_slice", "foi_topic",
]

[[deployment_group]]
slug = "group_foi_pipeline_cpu"
deployments = ["presidio", "question_slice_cpu", "question_extractor_tokenizer", "foi_topic"]
```

Call `POST /deployments/group_foi_pipeline/warm-up` with the usual authentication before a batch using GPU extraction. This warms Presidio, the local classification head and Granite tokenizer while creating or resuming both GPU deployments, even when automatic startup on inference requests is disabled. For CPU extraction, use `POST /deployments/group_foi_pipeline_cpu/warm-up`, which warms the full CPU classifier and the remote topic model. Existing per-deployment locks prevent overlapping group and inference requests from starting the same deployment twice within one API process.

Group warm-up returns HTTP 200 with `slug`, overall `success`, and an ordered `deployments` list. Each member has `slug`, `success`, `replicas` (null for local resources or on failure) and `error` (null on success). Inspect the success flags: a partial or complete startup failure is reported in the body. Unknown groups return 404. Successful members are not rolled back when another fails; retrying the group reuses loaded local resources and running deployments. Idle scaling and shutdown cleanup follow each resource's lifecycle, and warm-up does not keep a group running indefinitely.

Groups must be nonempty, have unique names, and reference local or Exoscale deployment slugs declared in the catalog without repeated members. Local and remote names must not collide. Catalog validation checks these references before registering cold factories, without importing resource owners or loading models. Adding a local deployment with a supported loader makes it available to individual and group warm-up without changing the HTTP handlers.

The `group_foi_response_pipeline` warm-up group includes `presidio`,
`foi_status_tokenizer`, and `foi_status`. Its tokenizer uses the same pinned
checkpoint reference as the remote deployment; `[foi.response]` configures the
total context and reserved output budget.

### Paid integration tests

Run `SERVER_ROLE=test poetry run pytest -m external -v` to exercise live deployments.
This uses stable names such as `clef_test`, `foi_status_test`, and `olmo3_7b_test`.
Toast tests are temporarily disabled with a module-level skip in `tests/test_toast.py`, even
when selected explicitly. Remove that skip only when Toast testing is needed again.
The Clef Compute test accepts `SERVER_ROLE=test` and deletes the test VM and access
resources. Keep `COMPUTE_STATE_DIR` persistent so interrupted runs can recover their
resource state. Sequential runs can reuse these names and the same state directory.
For concurrent isolated runs, use `SERVER_ROLE=test_<run>` and a separate
`COMPUTE_STATE_DIR` per run across all backends.

### Managed inference authentication

Set `EXOSCALE_INFERENCE_API_KEY` to a zone-scoped AI API key covering the managed
deployments used by this application. This is separate from the IAM
`EXOSCALE_API_KEY` and `EXOSCALE_API_SECRET` used for deployment management.
See [Exoscale’s current authentication guide](https://community.exoscale.com/product/concrete-ai/quick-start/).
The application uses that configured key for managed inference and never queries
Exoscale for deployment-specific credentials. Deployment status remains readable
without a key, but inference requires it.

Managed deployments can set `inference_engine_version` to select a provider runtime
explicitly. Omitting it keeps the provider default. OLMo selects `0.29.0`:
the provider default `0.31.0` crashes in Anthropic endpoint initialization when it
probes OLMo's template without defining `tools`. The 0.29.0 code path avoids that
probe when no explicit template override is supplied. See the upstream
[0.29.0 initialization](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/entrypoints/anthropic/serving.py)
and [0.31.0 probe](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/anthropic/serving.py).
Changing this pin requires a fresh deployment
and live inference checks.
