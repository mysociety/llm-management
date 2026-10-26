# Frozen authority research for model evaluations

These are real captures made through the configured model/search APIs and the
application's public-web reader. No page content or discovery hits were authored
as substitutes for the original sources. Each `capture.json` records the input,
UTC capture time, deployment slug, actual search results, extracted pages and
errors, and the capture run's finding and usage.

`expected.json` is a separate semantic review of those saved pages. Its SHA-256
pins the reviewed capture bytes. The capture model's result is historical output,
not an automatically generated golden answer. Different quotes are accepted when
supported by the captured material and the reviewed source URLs.

| Case | Reviewed route | Important distinction |
| --- | --- | --- |
| Croydon Council | `generic`: `contact@croydon.gov.uk` | No dedicated mailbox was established, but the saved customer commitment publishes this general address. Under the current fallback policy this is `found`, rather than `not_found`. |
| Abbey Lane Primary School, Sheffield | `generic`: `enquiries@abbeylane.sheffield.sch.uk` | Official general contact; the linked FOI policy PDF could not be read. |
| ARIA | `eir_only`: `EIR@aria.org.uk` | The saved page explicitly excludes FOI coverage and provides an EIR route. |
| East Dunbartonshire Council | `foi_eir`: `foi@eastdunbarton.gov.uk` | Official submission instructions and environmental-information guidance are captured. |

Some pages have truncated text or navigation lists, and unreadable PDFs remain
recorded failures. These are part of the research snapshot, not reasons to fetch
them again during a test. See each capture and its review notes for the limits.

## Compare models without scraping

```bash
script/test --all -m model_eval tests/test_foi_email_models.py \
  --foi-model gpt_6_1_low \
  --foi-model candidate_slug
```

Each slug must be an API model deployment in `conf/deployments.toml`. If no slug
is supplied, an explicitly selected model-evaluation run uses the finder’s
configured deployment. Ordinary `script/test` excludes these paid tests through
the `external` marker. Only the selected model service is contacted; no Tavily
key is needed for replay.

The normal agent and evidence validator run unchanged. A `ReplayResearch`
instance supplies both tool dependencies: every search receives the same
URL-deduplicated captured discovery set, regardless of wording, and page reads
return saved material by requested or final URL. An absent URL returns a fixture
error with no network fallback. This tests interpretation and research choices,
not live search ranking or current website contents.

Replay search usage is labelled `replay` with zero credits. Historical capture
charges remain in `capture.json`; they are not billed or counted again.

Reports under `.state/foi-email-evaluations/` retain case/model identity, fixture
checksum, latency, complete result and usage, requested search queries/pages,
missing URLs and failures. Use `--foi-eval-output PATH` to change the destination.
A missing URL is reported for diagnosis but does not itself fail a correct result.
The JSON reports are written before a failed semantic assertion is propagated.

## Capture another authority explicitly

Capture is an intentional live operation that can incur model/search charges
and fetch authority pages. It is never invoked by the evaluation tests:

```bash
poetry run python -m llm_management.foi.email_finder.capture \
  --name "Example authority" \
  --context "Public details that distinguish the authority" \
  --output .state/foi-email-captures/example/capture.json
```

Optional `--website URL` supplies an initial site; `--model SLUG` selects a capture
model. The capture uses the configured tool, request, timeout and download limits,
public-address transport, robots checks, user agent and host pacing. Existing
output files are protected before any paid calls. A partial failed run saves its
available material and exits unsuccessfully.

Review the saved pages, copy the capture into a new case directory here, and
create `expected.json` following the existing cases. Record accepted emails,
route type, evidence URLs and the reasoning, then compute `capture_sha256` from
the exact saved file. A new reviewed case directory is discovered automatically.
A directory without an expectation is skipped rather than captured or labelled
automatically. Updating a capture requires another review and checksum update.
