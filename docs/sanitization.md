# Sanitization and local resources

Both FOI pipelines run Presidio locally on CPU before model inference. The request
pipeline uses sanitized text for both CPU and Exoscale QuestionSlice and for Granite
classification, while API extraction results retain the original request text.
The response pipeline sanitizes request context, question text, additional text,
source bodies, and filenames together; question/source IDs and structural metadata
are preserved. The attested canonical input is passed to the configured remote
status model after the trained prompt has passed its token-budget check.

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
`question_slice` identifies the remote GPU deployment, `question_slice_cpu`
the full local CPU model, and `question_slice_head_cpu` its local classification
head used with GPU embeddings. Use these names in warm-up calls and groups.

Local resources load on demand. Use `POST /local-models/presidio/ensure`,
`POST /local-models/question_slice_cpu/ensure`,
`POST /local-models/question_slice_head_cpu/ensure`, or
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

Email validation uses the public-suffix snapshot bundled with `tldextract`. It does
not download suffix lists or write a suffix cache during sanitization.
