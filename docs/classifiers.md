# Correspondence classifiers

## SAR moderation on local CPU

Given an FOI request text, returns a quick indication of whether it is likely to be
a SAR, IMM, or other personal information request. If `fast` is true, returns only
logistic inference rather than cascading to the larger model.
Relevant deployment group: `group_sar_pipeline_cpu`.

`POST /endpoint/foi/sar_detection` accepts `{"body": "the complete original correspondence"}`
and an optional `fast` boolean (default `false`). It returns `is_sar`,
`stage` (`logistic` or `deberta`), `status` (`complete` or `unclear`), `reason`,
`logistic_score` and `deberta_score`. It flags requests for personal records,
including personal immigration correspondence and mixed personal/public requests;
it is not a general personal-information detector. This endpoint uses the usual
API authentication and does not start an Exoscale deployment or run the FOI pipeline.

The strict logistic model receives raw text. In default mode, every logistic positive proceeds to
DeBERTa using the shared predictor's placeholder removal, NFKC, casefold and
whitespace normalization. Only a successful negative can clear the flag. Loading,
inference, invalid-score, overlength and busy failures return HTTP 200 with
`is_sar: true`, `status: "unclear"` and a reason. Invalid HTTP request bodies still
receive normal validation errors. Treat unclear results as requiring moderator review.
Scores describe model outputs, not established real-world probabilities.

With `fast: true`, for example `{"body": "My personal records", "fast": true}`,
the endpoint returns the logistic decision immediately and skips DeBERTa for both
positive and negative results. It uses the existing inclusive cutoff
`0.0018603434604847564`; the pinned SAR release has no separate standalone policy.
Fast mode trades DeBERTa confirmation for speed and can retain additional SAR flags.
Both modes share the loaded logistic model. Logistic-only results use
`stage: "logistic"` and `deberta_score: null`; confirmation results use
`stage: "deberta"`. On failures, `stage` identifies the stage attempted, and both
modes retain the existing `is_sar: true`, `status: "unclear"` failure policy.

Models are lazy local resources named `sar_logistic_cpu` and
`sar_deberta_cpu`. Before a batch, call
`POST /deployments/group_sar_pipeline_cpu/warm-up`, or use individual
`POST /deployments/{name}/warm-up` calls. Inspect the group's member success flags.
Warm-up and loading are per API worker, and resources follow the normal CPU idle
unloading policy. Allow roughly 1.28 GB for the reported SAR process footprint,
plus other resources loaded in the same worker. DeBERTa uses CPU float32, the shared
`CPU_INFERENCE_THREADS` setting (default 1). Positive cascade decisions enter a
per-model async queue and are processed in batches of up to eight. Fast requests
and logistic negatives bypass that queue. Blocking inference runs off the event loop.
A full queue or expired result deadline returns an unclear `deberta_busy` result.

Private artifact access uses Pydantic Settings: `HF_TOKEN` takes precedence over
legacy `HUGGINGFACE_TOKEN`. `CLASSIFIER_CACHE_DIR` controls the Hugging Face cache;
mount it persistently in containers to avoid repeat downloads. Tokens and model
weights are not included in the repository. Artifact identities are pinned in
`conf/deployments.toml`; trained cutoff checks remain in the detector implementation:

| Model | Commit | Inclusive positive cutoff |
|---|---|---|
| `mySociety/sar-detect-logistic-2026-10-09-1` | `ade6333a06995cbde835648d758d8b092900bc52` | JSON cutoff `0.0018603434604847564` |
| `mySociety/sar-detect-deberta-2026-10-09-1` | `199883a0387af896f34f7bf0978472268a9474b9` | Softmax index 1, `0.5` |

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


## Automatic-response classification on local CPU

Given a response to an FOI request, returns a quick indication of whether it is
likely to be an automatic response. If `fast` is true, returns only logistic
inference rather than cascading to the larger model.
Relevant deployment group: `group_automatic_pipeline_cpu`.

`POST /endpoint/foi/automatic_response_detection` accepts one current email body:

```json
{"body": "This is an automatic reply. I am out of the office.", "fast": false}
```

The endpoint uses the usual API authentication and returns `is_automatic`, `stage`
(`logistic`, `deberta`, or `none`), `unresolved`, `reason`, `logistic_score`, and
`encoder_score`. It reports classification only; callers decide what action to take.
A resolved result has `unresolved: false` and `reason: null`. `stage` identifies the
last stage attempted, including a stage that failed. Scores are model outputs,
not established real-world probabilities.

By default, the logistic `model.json` policy gates at the inclusive cutoff
`0.4292739638892871`. A negative finishes at `stage: "logistic"`. A positive goes to
CPU DeBERTa-v3-base in PyTorch float32, with labels `0=authored`, `1=automatic`.
Only a positive softmax score of at least `0.5` confirms `is_automatic: true`.
With `fast: true`, the separately packaged `standalone-model.json` policy determines
the result and DeBERTa is skipped. Its cutoff currently matches the cascade gate;
its policy and the cascade policy share the same loaded logistic weights.

Empty bodies (including whitespace or placeholders only), inference errors, invalid
scores, and encoder inputs above 512 tokens including special tokens return HTTP 200
with `is_automatic: false`, `unresolved: true`, and a reason. Encoder inputs are never
truncated. Reasons include `empty_body`, `logistic_failure`, `overlong_body`,
`encoder_failure`, `encoder_unavailable`, `encoder_busy`, `invalid_score`,
`inference_failed`. Queue overload or deadline expiry uses `encoder_busy`.
An unresolved result must be distinguished from a resolved authored classification.
Malformed HTTP request bodies receive normal validation errors.

Models are lazy local resources, loaded on demand and shared per API worker.
Before a batch, call `POST /deployments/group_automatic_pipeline_cpu/warm-up`, or
warm `auto_logistic_cpu` and `auto_deberta_cpu` individually with
`POST /deployments/{name}/warm-up`, just as for SAR. Inspect the warm-up group's
member success flags. Missing models or credentials fail warm-up; inference-time
loading failures return unresolved classification results. API startup does not
load these models or require their downloads to succeed.
Both follow `CPU_IDLE_TIMEOUT_MINUTES` (default 15) and reload on subsequent use.
Fast calls and cascade negatives only load the logistic model; DeBERTa loads when
confirmation is needed or explicitly warmed. Loaded logistic weights are shared
between modes until idle unloading, and model leases protect active worker use.
Private downloads use `HF_TOKEN` (or legacy `HUGGINGFACE_TOKEN`) and
`CLASSIFIER_CACHE_DIR`. Loading validates both logistic JSON checksums against the
supplied release manifest. Revisions are pinned in `conf/deployments.toml`:

| Model | Commit |
|---|---|
| `mySociety/autoreply-logistic-2026-10-09-1` | `9cff7b1018a61bd919fb049e3aca005952a101c7` |
| `mySociety/autoreply-deberta-2026-10-09-1` | `ddf70e25af71b64260e72ac3cdef1353a990d4c8` |

Blocking inference runs off the async event loop. `AUTOMATIC_MAX_CONCURRENCY`
(default 4) limits concurrent logistic worker calls; further calls await capacity
without occupying a worker thread. Positive cascade decisions then enter the
model's async batch queue. Fast mode uses its own logistic decision policy and
never joins the DeBERTa queue. Cancellation retains logistic capacity and model
resource leases until already-running work finishes.

### Queued DeBERTa inference

SAR and automatic-response detection each have an independent queue per API worker:

| Setting | Default | Meaning |
|---|---:|---|
| `DEBERTA_BATCH_SIZE` | 8 | Maximum documents in one inference batch |
| `DEBERTA_BATCH_DELAY_MS` | 10 | Maximum collection delay for a partial batch; full batches start immediately when the model is free |
| `DEBERTA_QUEUE_SIZE` | 64 | Maximum waiting documents, in addition to the active batch |
| `DEBERTA_QUEUE_TIMEOUT_SECONDS` | 30 | Deadline from enqueue to result, including model loading and inference |
| `CPU_INFERENCE_THREADS` | 1 | Shared process-wide PyTorch thread count |

A queued single request waits up to 10 ms to collect peers before model execution
when idle. Under load, it also waits for earlier batches. Queue waiting does not
occupy inference threads. Invalid or overlong inputs are isolated from valid peers;
inputs remain untruncated. Cancelled or expired queued requests are skipped. If
inference has already started, it finishes under its model lease even when the
caller disconnects or times out. Server shutdown rejects pending work and waits
for the active batch before unloading resources. Group cool-down and scale-to-zero
continue to protect active inference; queued work alone does not reserve a model.

The synchronous Python classifier helpers still run directly; cross-request batching
is provided by the async service and HTTP endpoints. Each API worker owns its own
queues and model copies. This batching change does not apply to QuestionSlice.

The generic logistic predictor and authenticated tokenizer loader are shared with
SAR. Normalization removes angle-bracket placeholders, applies NFKC and casefold,
and collapses whitespace. As with SAR, the service keeps its existing Python
runtime and intentionally relaxes upstream's Python 3.14 / Unicode 16.0.0 runtime
check. Model metadata must still declare Unicode 16.0.0; the host's Unicode tables
can differ for some characters, so this adaptation does not establish full parity.
The adapted `AutoDetector` follows the supplied upstream `inference.py`, using
service-owned models and classification-only results.

The reported synthetic test classified all 133 automatic and 130 authored replies
correctly with the cascade. Logistic alone incorrectly classified one authored
reply as automatic. Reported benchmark CPU timings were approximately 0.11 ms
median for logistic and 151 ms average for the cascade; these are not service SLAs.
Real-email validation remains necessary before callers enable notification suppression.
