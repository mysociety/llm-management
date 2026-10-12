# Proposal: request extraction on shared GPU Compute

Design notes from 12 October 2026. This is a proposal; no infrastructure or
application changes have been implemented for it.

## Current setup

The FOI pipeline uses two checkpoints:

- QuestionSlice / ModernBERT identifies the question structure. CPU mode runs
  the complete classifier locally. The GPU path uses a managed Exoscale
  embeddings deployment, with the original classification head running in the API.
- Fine-tuned Granite 1B generates topic classifications through a separate
  managed Exoscale deployment. This remote stage is also used with CPU extraction.

Both managed deployments currently request one A5000 GPU each. Exoscale's
Dedicated Inference API accepts one model per deployment and exposes no
documented mechanism for placing these two deployments on one GPU.

An Exoscale GPU Compute VM gives us control over the serving processes, allowing
both models to share a GPU. An H100 should have ample capacity in principle, but
we should measure serving memory and throughput before choosing hardware; a
smaller GPU may suffice.

## Proposed serving setup

Use a reusable vLLM runtime configured with a list of model services. Distinguish
four artifacts:

- **Container image:** a pinned, tested vLLM runtime, reusable across checkpoints.
- **Base model checkpoint:** weights and configuration pinned to a model revision.
- **LoRA adapter:** optional fine-tuning weights pinned separately and attached to
  a compatible base checkpoint.
- **Exoscale VM template:** a prepared machine image with the host runtime and,
  optionally, the chosen checkpoints cached for fast startup.

Run one vLLM container per base checkpoint, using the same container image and GPU,
with separate listening ports and explicit GPU memory budgets.

```text
One Exoscale GPU Compute VM
├── vLLM service: QuestionSlice / ModernBERT → embeddings
└── vLLM service: Granite 1B + FOI LoRA    → chat completions
```

ModernBERT retains its pooling/embeddings serving arguments. Granite retains its
text-generation and structured-output arguments, with the option to load the
FOI adapter separately instead of using the merged checkpoint. Initially, keep the existing
QuestionSlice classification head in the API to minimise pipeline changes.

A model-bundle recipe should declare, for each service:

- Service name and model reference/revision.
- Container image reference, defaulting to the shared pinned vLLM image.
- Listening port and expected API/task.
- Serving arguments, including context limits and precision.
- GPU memory budget, including adapter capacity where enabled.
- Optional named adapters, repository revisions, local artifact paths, maximum
  LoRA rank and maximum adapters allowed per batch.
- Readiness checks and representative smoke tests.

This uses multiple vLLM processes on one GPU, rather than assuming one vLLM
process can load both unrelated checkpoints. vLLM documents per-instance memory
budgets for sharing a GPU.

## LoRA support

The reusable vLLM container image must support LoRA serving. Compatibility also
needs validation for each base model architecture and adapter's target modules;
a general LoRA-capable runtime does not guarantee every checkpoint/adapter pair
will work. Keep LoRA optional for each model service.

For Granite, load one base model and register the FOI adapter at startup using
`--enable-lora`, `--lora-modules` and `--max-lora-rank=16`. Select the adapter by
its stable serving name in the request's `model` field. Additional compatible
adapters could share that same Granite process and base weights. Unrelated base
models, such as ModernBERT and Granite, still use separate processes.

The first adapter pair to validate is:

| Artifact | Repository | Pinned revision |
| --- | --- | --- |
| Base | `ibm-granite/granite-4.0-1b` | `6a7381ba1f54d684ff508d991aeb7dc580157103` |
| FOI adapter | `mySociety/granite-tiny-foi-topic-grounded-v2` | `200b756850c137c255f2e6c5c24474edd894d010` |

Authenticated metadata inspection confirmed the adapter is private, has rank 16
and alpha 32, and targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `input_linear`
and `output_linear`. It has no `modules_to_save`, uses no DoRA, and specifies
`bias=none`. The service must support all of its actual adapted layers.

Download adapters at their pinned revisions during authenticated preparation,
then load them from local paths. Cache them alongside the base checkpoint in the
prepared template without embedding download credentials. Track tokenizer and
chat-template choices explicitly and preserve the existing FOI prompting.

Start with a fixed adapter list registered at startup. Runtime adapter upload,
replacement and unloading are deferred; initial support does not require enabling
the dynamic LoRA administration endpoints.

Validation should establish that the adapter is registered, selected and applied,
not merely that the server starts. Compare base-only, adapter-selected and current
merged-checkpoint outputs on representative FOI prompts with deterministic
settings, allowing for numerical differences between serving implementations.
Repeat readiness and adapter checks after VM recreation. Account for adapter
buffers in the shared GPU memory budget and protect requests to all adapter
names with the same VM request leases.

### Confirmed hosted-inference limitation

On 12 October 2026, a real Dedicated Inference create request in `at-vie-2`
was rejected with HTTP 400 before deployment creation:

```text
Parameter 'enable-lora' is not allowed for vLLM 0.31.0
Parameter 'lora-modules' is not allowed for vLLM 0.31.0
Parameter 'max-lora-rank' is not allowed for vLLM 0.31.0
```

The probe used the already imported merged Granite checkpoint to test argument
validation. It did not test base-plus-adapter compatibility, and no deployment
was created. This confirms a restriction in Exoscale's hosted API; it does not
establish a missing feature in upstream vLLM. GPU Compute gives us control over
the serving flags, but the Granite adapter still needs an actual compatibility
and inference test there.

## Host preparation and model bundles

Keep the reusable host setup separate from the model bundle:

- Host setup installs NVIDIA drivers, Docker/GPU container support and caches
  the pinned serving image.
- Bundle preparation downloads the configured base and adapter revisions and validates
  both services before creating a prepared Exoscale template.
- Runtime provisioning starts the configured services and waits for readiness.

Changing checkpoints should not require rebuilding the serving container image.
A prepared template can still contain a particular bundle's cached checkpoints,
so changing that bundle may require a new template release.

Our existing Compute lifecycle for Clef provides provisioning, templates, access
resources, tunnels and teardown. Its current template recipe supports only Clef;
we would extend the recipe/bootstrap/probe support for vLLM services and multiple
endpoints. Clef can remain a separate recipe using its specialised container.

## Application lifecycle

Represent the shared VM as one deployment that owns multiple model endpoints.
Keep model identities and endpoint routing separate from VM ownership.

- Ensuring either model starts or reuses the shared VM.
- Requests to either service refresh the shared idle timer.
- In-flight requests to either service protect the VM from teardown.
- Cooldown and scale-to-zero apply to the entire bundle.
- Readiness/status should expose each service's state as well as VM state.
- Shutdown deletes the VM and its owned access resources once, rather than once
  per model endpoint.

The FOI pipeline can continue to choose extraction and topic endpoints by name,
with both resolving to services on the same deployment. Separate ports are enough
for the initial design; a shared routing proxy is optional.

## Initial scope and open questions

Start with a fixed bundle and fixed adapter list selected before startup, and
keep both base models resident.
Dynamic model addition/removal or sleep/wake scheduling can be considered later.

Before implementation or deployment, resolve:

- Which GPU is sufficient and available in the chosen Exoscale zone?
- Which pinned vLLM version supports both checkpoints, the complete Granite FOI
  adapter and existing serving flags?
- How much memory should each service reserve, including weights, runtime
  overhead, activations and Granite's generation cache?
- What throughput and latency do we get when the services share a GPU?
- How should service names map to their shared deployment in the catalog?
- How should startup failure of one service affect bundle readiness and cleanup?

Do not use the default large vLLM memory allocations independently for both
servers. Measure explicit budgets under realistic context lengths, batch sizes
and concurrent requests. Weight size alone does not establish serving capacity.

## References

- [Exoscale Dedicated Inference overview](https://community.exoscale.com/product/concrete-ai/dedicated-inference/overview/)
- [Exoscale deployment API](https://community.exoscale.com/reference/api/ai/dedicated-inference/deployment/)
- [Running inference on Exoscale GPU Compute](https://community.exoscale.com/product/compute/instances/how-to/running-llms-on-gpus/)
- [vLLM Docker serving image](https://docs.vllm.ai/en/latest/deployment/docker/)
- [vLLM per-instance GPU memory allocation](https://docs.vllm.ai/en/latest/cli/serve/#--gpu-memory-utilization--device-memory-utilization)
- [vLLM LoRA serving and adapter selection](https://docs.vllm.ai/en/latest/features/lora/)
