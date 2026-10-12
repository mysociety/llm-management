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
