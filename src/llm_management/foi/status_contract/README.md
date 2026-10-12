# FOI response-status model contract

These are pinned assets for the trained response-status model, shipped in the
Python package so inference does not need to fetch instructions at runtime.

- `SYSTEM-PROMPT.txt` is read by `response_model.prepare_request()` and sent as
  the system message.
- `OUTPUT-SCHEMA.json` is read by `response_model.output_schema()` and sent as
  the structured-output schema.
- `TASK-INSTRUCTION.txt` is a reference copy of the published task instruction;
  it is not loaded separately at runtime.
- `PROVENANCE.json` records the source repository, revision and asset hashes.
  It is reference metadata, not runtime configuration.

The runtime verifies the system prompt and output schema against hashes in
`response_model.py`. Update assets and pins together when intentionally changing
the model contract. The public API's enriched request questions are projected
into the model's input schema; they do not require changing these trained assets.
