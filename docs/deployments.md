# Deployment management and warm-up

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

To request a replica count, call `POST /deployments/{slug}/ensure?replicas=3`.
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

Call `POST /deployments/{slug}/cooldown` after a batch to reduce the remaining
idle countdown to at most 60 seconds. A shorter countdown is preserved. The
response includes `seconds_until_scale_to_zero`; already stopped deployments
return `already_stopped` with a null countdown. Active requests return HTTP 409.
Subsequent traffic or ensure calls reset the normal idle timer. The background
scaler checks every 60 seconds, so shutdown can occur up to one minute after
the shortened countdown expires.

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
