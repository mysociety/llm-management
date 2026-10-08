# Preparing Exoscale model servers

A prepared Exoscale GPU VM template contains the serving container, GPU host
software and cached model weights. Starting a server then requires booting the
VM and loading the model into GPU memory, with no image pull or model download.

Template preparation and Compute deployment management live in the main package
and CLI, alongside Exoscale managed inference. The `clef` deployment uses the
`clef_flash` template recipe.

## Container and template releases

The [System One container repository](https://github.com/mysociety/systemone-container/)
owns the Dockerfile and GitHub Action that publishes the image. There is no local
copy in this repository. The container selects its model at runtime using
`SYSTEMONE_BACKEND`, `SYSTEMONE_MODEL` and `SYSTEMONE_REVISION`. Currently only
Clef is supported; another family needs a loader and possibly new dependencies.

An Exoscale template contains that image plus a specific cached model revision
and boot service. Use a new versioned template name when changing the container,
model revision or host setup. Exoscale assigns a UUID, which our configuration
resolves from the private template's name in its zone. Missing or ambiguous
matches fail explicitly.

## Template configuration

[conf/exoscale_templates.toml](conf/exoscale_templates.toml) contains named
recipes, separately from deployment configuration:

```toml
[[template]]
slug = "clef_flash"
name = "systemone-clef-flash-2026-10-08-v1"
zone = "at-vie-2"
image = "ghcr.io/mysociety/systemone-container:sha-b783a1e9a83fc3ef5cfca97c2a374b7cc4dbe2fe"
backend = "clef"
model = "Cloudflare/clef-flash"
revision = "fde727a287004204b7518dcc983fe64379776712"
max_length = 4096
instance_type = "gpua5000.small"
disk_size_gib = 100
```

Each recipe also requires at least one `[[template.smoke]]` choice question,
with `state`, `instructions`, `criteria` and `expected`. See the checked-in
recipe for complete examples. These cases verify the model over HTTP, followed
by a native Pydantic AI request using the first case. Health must report the
configured backend, model, revision and context limit.

The default challenges ask for France's capital and the result of two plus two.

Optional fields include `base_template_name` (Ubuntu 24.04 by default),
`nvidia_driver_package`, `port`, `ssh_cidr`, `startup_timeout` (1,800 seconds),
`poll_interval` (five seconds) and `description`. The host recipe currently
assumes Ubuntu with apt, Docker and NVIDIA GPUs. Template names are shared
release artifacts and do not acquire a deployment `server_role` suffix.

The container image is resolved to its registry digest during preparation.
For fully pinned inputs, put an `@sha256:...` image reference in the recipe.
The model revision must be a full commit hash. Preparation records the recipe
and resolved template UUID in its report, and stores the image digest on the VM.

## CLI workflow

Install dependencies with `poetry install`, set `EXOSCALE_API_KEY` and
`EXOSCALE_API_SECRET` in the environment or `.env`, and have `ssh` and
`ssh-keygen` installed locally.

```sh
# Show recipes and their registered template UUIDs, where present.
llm-management templates list

# Prepare a new release and verify two fresh VMs.
llm-management templates create clef_flash --runs 2

# Resolve the configured private template name to its UUID.
llm-management templates resolve clef_flash

# Test an existing release without rebuilding it.
llm-management templates test clef_flash --runs 2
```

Use `poetry run` before these commands when working in Poetry's environment.
An alternate configuration file can be selected with
`llm-management templates --config path/to/recipes.toml ...`.

`create` refuses to replace an existing named template. Choose a new versioned
name for a new release. The checked-in name was registered through this CLI on 8 October 2026.
To select a different retained release, change the recipe's `name` and keep its
model and runtime settings consistent with that release.

Both `create` and `test` incur cloud charges. Only SSH is exposed, restricted to
`--ssh-cidr` or the recipe's `ssh_cidr`; otherwise the caller's public IPv4 is
discovered. HTTP is bound to localhost and reached through an SSH tunnel.

## What preparation does

1. Create a disposable GPU builder VM using the recipe's OS template, instance
   type and disk size.
2. Install Docker, the NVIDIA driver and NVIDIA Container Toolkit. Verify a CUDA
   tensor operation inside the container.
3. Pull the image and save its digest in `/opt/systemone/image`.
4. Run the configured model online once, caching the pinned revision under
   `/opt/systemone/model-cache`. Verify health and the configured smoke cases.
5. Enable `systemone.service` to use the cached image with `--pull never` and
   `HF_HUB_OFFLINE=1`. Its model cache is mounted at `/data/huggingface` and its
   runtime settings are saved in `/opt/systemone/runtime.env`.
6. Stop the server, remove builder SSH access and host keys, reset cloud-init
   and machine identity, then stop the VM. Snapshot and export the disk, and
   register a private template with the configured name in the selected zone.
7. Delete temporary builder resources. Boot fresh VMs from the template, repeat
   the checks, and delete their temporary resources too.

Logs, state and the report go under the printed `/tmp/llm-template-...` directory.
Cleanup runs on success, failure and Ctrl-C. If a process is killed or cleanup
fails, recover the affected stage using its manifest:

```sh
llm-management templates cleanup /tmp/llm-template-.../builder/state.json
```

Temporary snapshots, VMs, SSH keys and security groups are removed. The template
is retained even if subsequent smoke tests fail, so inspect the report before
using a new release. Retired templates must be deleted separately. Template
storage is charged even when no VMs exist, based on virtual disk size; see
[Exoscale pricing](https://www.exoscale.com/pricing/).

## Deploying a prepared server

Deployment entries in `conf/exoscale.toml` reference a recipe slug. The recipe
supplies the template name, zone, model and VM settings. The checked-in Clef
entry is:

```toml
[[deployment]]
slug = "clef"
backend = "exoscale_compute"
protocol = "systemone"
template = "clef_flash"
```

Existing managed entries default to `backend = "exoscale_managed"` and
`protocol = "openai"`. Infrastructure backend and API protocol are separate from
the template recipe's model-family backend (`clef`).

The checked-in recipe points to the verified named template above. For a new
release, prepare its named template first. Then use the normal deployment commands:

```sh
llm-management create-or-resume clef
llm-management logs clef
llm-management pause clef
```

The same lifecycle is used by `/deployments/clef/ensure` and requests to
`/v1/systemone` or `/agents/immigration_detection/clef`. Startup resolves the
private template name to its UUID, creates one GPU VM, opens an SSH tunnel and
waits for matching health before forwarding requests. Concurrent requests share
the startup lock. An existing stopped VM is resumed; after a manager restart,
its VM is rediscovered and the tunnel is recreated.

Only SSH is exposed, restricted to the recipe's caller CIDR. HTTP remains on
localhost; the manager forwards over the tunnel without an upstream bearer key.
The manager's own client authentication remains in place. OpenAI agent endpoints
reject System One deployments.

Persist `COMPUTE_STATE_DIR` (default `.state/compute`) across manager restarts.
It contains per-zone, per-deployment manifests and private SSH keys. Use one
manager process/worker per deployment. Local filesystem locks coordinate CLI and
server lifecycle operations; request counters belong to the server process.
Resources are named `llm-compute-{zone}-{slug}_{server_role}`, keeping zones, test
and production separate.

For Compute, `replicas = 1` in API state means the model is ready through this
manager's tunnel; a discovered VM with no ready connection reports zero. The
`connect` command includes an SSH command for opening a tunnel yourself. Its
reported temporary localhost URL belongs to that CLI process and closes when
the process exits.

Idle timeout, `pause`, `destroy`, `/scale-to-zero` and server shutdown delete
Compute VMs and their SSH keys/security groups, and close tunnels. Managed
inference continues scaling to zero. Active HTTP requests prevent idle teardown and cause the HTTP
`/scale-to-zero` operation to return 409. CLI teardown is an administrative action. New VMs that fail startup are cleaned up; if cleanup
fails, the durable manifest and logs support retrying `destroy clef`. An existing
VM that fails reconnection remains discoverable for idle cleanup or explicit
teardown. Templates remain until separately retired.
