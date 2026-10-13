# FOI live endpoint validation examples

Ten copied validation examples for an agent testing the application API: five raw requests with expected extraction and topic-classification results, and five response-analysis inputs with expected outputs. These are dataset expectations, not outputs observed from a live endpoint. No live inference was run to prepare this file.

## How to run and compare

- Use the app’s base URL and configured authentication, with `Content-Type: application/json`. Send only the **Input** JSON block for each case; case metadata is not part of the HTTP body.
- For R1–R5, POST to `/endpoint/foi/extract_questions/extract?backend=cpu` (or `backend=exoscale` for remote extraction). To test extraction plus topic classification, send the same body to `/endpoint/foi/extract_questions?backend=cpu` (or `backend=exoscale`) and also compare the **Expected full-endpoint fields** block for that case. That block shows the merged public question fields and request topics. It is not another input payload.
- For S1–S5, POST to `/endpoint/foi/response_status`. Each input already includes its own gold request context; do not replace it with R1–R5 or with fresh model predictions when testing response analysis in isolation.
- Require HTTP 200, then compare parsed JSON. R1–R5 expectations are a subset of the full response: compare `questions`, `extraction_status` and `additional_text`, retaining question order and IDs. For the full `/endpoint/foi/extract_questions` endpoint, compare each question’s `regime` and `topic` against the full-endpoint block, and compare top-level `request_topics`. Ignore other diagnostic/provenance fields. S1–S5 expectations are the complete public output, including `process_states: null`.
- Start with exact field comparisons; report whitespace-only text differences separately from changed wording, missing content or different question grouping. JSON object key order does not matter. For response semantic comparison, outcomes/events may be matched by their full contents irrespective of array order; retain scope IDs, acts, visibility, source references and legal references, and reject missing or extra items.
- For topic classification, require exact question IDs, order and regimes. Record exact topic-string matches against the dataset separately from semantically equivalent wording; changed subject matter is a semantic mismatch. If extracted question text or grouping differs, flag the extraction mismatch first because classification depends on those questions.
- JSON `\n` escapes represent the original newlines: parse the JSON once and send the resulting strings without rewriting bullets, line breaks, placeholders or relative dates.
- Optional warm-up: POST `/deployment-groups/foi_pipeline_cpu/ensure` for the CPU request pipeline, `/deployment-groups/foi_pipeline/ensure` for the remote request pipeline, or `/deployment-groups/foi_response_pipeline/ensure` for responses. Check the returned success flags before submitting the batch. Warm-up and inference can start configured deployments.
- Record case ID, endpoint/backend, HTTP status, returned model provenance where available, latency, actual output and differences from the expected fields. Report transport/schema failures separately from semantic mismatches.

## Dataset provenance

Request labels come from [mySociety/foi-modernbert-question-slice-v2](https://huggingface.co/datasets/mySociety/foi-modernbert-question-slice-v2/tree/5f3d5085f3e64398df248e6009f214a5eda52c3c), revision `5f3d5085f3e64398df248e6009f214a5eda52c3c`, split `validation_heldout_topic.jsonl`. That export contains labelled semantic units rather than complete raw messages. Full request text was copied without rewriting from the matching `case_id` in [mySociety/foi-granite-topic-grounded-v2](https://huggingface.co/datasets/mySociety/foi-granite-topic-grounded-v2/tree/d52dfb7def21f967d914066434438a351afcc985), revision `d52dfb7def21f967d914066434438a351afcc985`, the same split. All five expected question lists were independently reconstructed from the QuestionSlice gold labels and checked against that companion export. `ADDITIONAL` units form the expected additional_text lists. Topic-classification expectations are copied verbatim from `completion[0].content` in those same companion topic records and merged with reconstructed question text by ID in the full-endpoint blocks below. Their source revision and file checksum are the same as for the raw request text.

The five request examples have distinct `case_id`/`group_id` values and different subjects/structures. These exports do not expose a separate `family_id`; descriptive family headings below are editorial categories, not claimed dataset taxonomy. “Raw” means the complete pre-extraction text supplied by the dataset, including its existing anonymization placeholders.

Response inputs and gold outputs come from [mySociety/foi-response-extraction-v4-synthetic-v3](https://huggingface.co/datasets/mySociety/foi-response-extraction-v4-synthetic-v3/tree/ddd3b5476a2b21b6800fe8b7e6d1d2269ef3fb8b), revision `ddd3b5476a2b21b6800fe8b7e6d1d2269ef3fb8b`, split `validation.jsonl`. This is the dataset revision identified by the configured status model’s card. The five examples have distinct dataset `family_id` values and different recipes. Inputs are copied from `input_json`; expected outputs are copied from `output_json` with only `process_states: null` added to match the public API’s default. Source availability, IDs, prior/current roles and text are preserved.

These Hugging Face repositories require appropriate access to inspect the originals; all ten payload pairs are self-contained below. The response split is synthetic development data, with candidate/unreviewed provenance; use its labels as regression expectations, not independent evidence of real-world accuracy. Five cases per endpoint are a small smoke-test sample.

All source line numbers below are 1-based JSONL line numbers.

## Request extraction and topic classification: five cases

### R1 — Multiple prose questions: transport budgets and policing

- `case_id` / `group_id`: `e39048d3-3c1a-5d89-9690-13205d79aa4f`
- Raw-text source: companion topic split line 1, `prompt[1].content.request_text`.
- Gold-label source: QuestionSlice split lines 1–6.

Three separate questions; budget qualifications remain attached to the first two questions.

**Input**

```json
{
  "body": "I am writing to request some information about how public funds are allocated and managed in our region, particularly regarding transport and policing. Could you provide a breakdown of how the regional transport budget is allocated to different public transport modes, such as bus, rail, and trams? I am interested in the most recent summary of budget allocations and any available figures for the current financial year. Additionally, I would like to know the annual budgets and expenditure for highway maintenance for each local authority district within the region. Please include winter gritting expenditure figures if available. Finally, I would appreciate details about the legal requirements and process that guide police stop and search procedures in this region."
}
```

**Expected extraction fields**

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "Could you provide a breakdown of how the regional transport budget is allocated to different public transport modes, such as bus, rail, and trams? I am interested in the most recent summary of budget allocations and any available figures for the current financial year."
    },
    {
      "question_id": "q2",
      "text": "Additionally, I would like to know the annual budgets and expenditure for highway maintenance for each local authority district within the region. Please include winter gritting expenditure figures if available."
    },
    {
      "question_id": "q3",
      "text": "Finally, I would appreciate details about the legal requirements and process that guide police stop and search procedures in this region."
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I am writing to request some information about how public funds are allocated and managed in our region, particularly regarding transport and policing."
  ]
}
```

**Expected full-endpoint fields**

Source: companion topic split line 1, `completion[0].content`. Question text and classification are merged by question ID. Runtime metadata and original request text are omitted from this comparison.

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "Could you provide a breakdown of how the regional transport budget is allocated to different public transport modes, such as bus, rail, and trams? I am interested in the most recent summary of budget allocations and any available figures for the current financial year.",
      "regime": "FOI",
      "topic": "regional transport budget allocation by public transport mode"
    },
    {
      "question_id": "q2",
      "text": "Additionally, I would like to know the annual budgets and expenditure for highway maintenance for each local authority district within the region. Please include winter gritting expenditure figures if available.",
      "regime": "FOI",
      "topic": "highway maintenance budgets and expenditure by local authority district"
    },
    {
      "question_id": "q3",
      "text": "Finally, I would appreciate details about the legal requirements and process that guide police stop and search procedures in this region.",
      "regime": "FOI",
      "topic": "police stop and search legal requirements"
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I am writing to request some information about how public funds are allocated and managed in our region, particularly regarding transport and policing."
  ],
  "request_topics": [
    "regional transport budget allocation by public transport mode",
    "highway maintenance budgets and expenditure by local authority district",
    "police stop and search legal requirements"
  ]
}
```

### R2 — Wrapped lines and a list item: planning statistics

- `case_id` / `group_id`: `e565022d-c595-5bc2-8026-734024b8ab62`
- Raw-text source: companion topic split line 3, `prompt[1].content.request_text`.
- Gold-label source: QuestionSlice split lines 12–17.

Join wrapped lines and keep the requested monthly breakdown within q1. Existing redaction placeholders are literal dataset text.

**Input**

```json
{
  "body": "I am writing to request a statistical overview of planning\napplications handled by your authority. This request is made\nunder the relevant provisions for access to recorded\ninformation held by public authorities.\nRequester: <PERSON>. Email: <EMAIL_ADDRESS>.\n- Please provide the number of planning applications received and the number determined each month for the period from January 2023 to December 2023.\nThe figures should be broken down by calendar month and\nindicate the number of applications received, the number\napproved, refused, withdrawn, and those still under\nconsideration at year end."
}
```

**Expected extraction fields**

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "- Please provide the number of planning applications received and the number determined each month for the period from January 2023 to December 2023. The figures should be broken down by calendar month and indicate the number of applications received, the number approved, refused, withdrawn, and those still under consideration at year end."
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I am writing to request a statistical overview of planning applications handled by your authority.",
    "This request is made under the relevant provisions for access to recorded information held by public authorities.",
    "Requester: <PERSON>.",
    "Email: <EMAIL_ADDRESS>."
  ]
}
```

**Expected full-endpoint fields**

Source: companion topic split line 3, `completion[0].content`. Question text and classification are merged by question ID. Runtime metadata and original request text are omitted from this comparison.

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "- Please provide the number of planning applications received and the number determined each month for the period from January 2023 to December 2023. The figures should be broken down by calendar month and indicate the number of applications received, the number approved, refused, withdrawn, and those still under consideration at year end.",
      "regime": "FOI",
      "topic": "planning application statistics"
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I am writing to request a statistical overview of planning applications handled by your authority.",
    "This request is made under the relevant provisions for access to recorded information held by public authorities.",
    "Requester: <PERSON>.",
    "Email: <EMAIL_ADDRESS>."
  ],
  "request_topics": [
    "planning application statistics"
  ]
}
```

### R3 — Environmental request with separately labelled context: air monitoring

- `case_id` / `group_id`: `bada42e1-214b-53c7-8f4d-a43a618b83e6`
- Raw-text source: companion topic split line 6, `prompt[1].content.request_text`.
- Gold-label source: QuestionSlice split lines 27–29.

The dataset labels the two sentences beginning “Focus on” and “Specify” as ADDITIONAL. Preserve them in additional_text; this is the recorded gold expectation, even though they could plausibly be interpreted as question qualifications.

**Input**

```json
{
  "body": "• List the geographic locations and operational status of all active sites in the national air quality monitoring network.\nFocus on sites that currently measure PM2.5, NO2, O3, SO2, and CO in line with UK-adopted EU directives. Specify whether each site is urban, rural, or roadside."
}
```

**Expected extraction fields**

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "• List the geographic locations and operational status of all active sites in the national air quality monitoring network."
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "Focus on sites that currently measure PM2.5, NO2, O3, SO2, and CO in line with UK-adopted EU directives.",
    "Specify whether each site is urban, rural, or roadside."
  ]
}
```

**Expected full-endpoint fields**

Source: companion topic split line 6, `completion[0].content`. Question text and classification are merged by question ID. Runtime metadata and original request text are omitted from this comparison.

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "• List the geographic locations and operational status of all active sites in the national air quality monitoring network.",
      "regime": "EIR",
      "topic": "national air quality monitoring network sites"
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "Focus on sites that currently measure PM2.5, NO2, O3, SO2, and CO in line with UK-adopted EU directives.",
    "Specify whether each site is urban, rural, or roadside."
  ],
  "request_topics": [
    "national air quality monitoring network sites"
  ]
}
```

### R4 — Personal records: social care correspondence

- `case_id` / `group_id`: `ea48f5fc-fa19-595d-bfdd-978cd3e2fe0e`
- Raw-text source: companion topic split line 14, `prompt[1].content.request_text`.
- Gold-label source: QuestionSlice split lines 63–67.

The repeated introductory request is ADDITIONAL. Extract the explicit request and its date qualification as one question.

**Input**

```json
{
  "body": "I am writing to request copies of all correspondence between\nmyself and my allocated social care support worker.\nThis request is from <PERSON>. Replies can be sent to\n<EMAIL_ADDRESS>.\nPlease provide copies of all emails or letters exchanged\nbetween me and my allocated social care support worker.\nThese records should cover the period from January 2023 to\nthe present."
}
```

**Expected extraction fields**

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "Please provide copies of all emails or letters exchanged between me and my allocated social care support worker. These records should cover the period from January 2023 to the present."
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I am writing to request copies of all correspondence between myself and my allocated social care support worker.",
    "This request is from <PERSON>.",
    "Replies can be sent to <EMAIL_ADDRESS>."
  ]
}
```

**Expected full-endpoint fields**

Source: companion topic split line 14, `completion[0].content`. Question text and classification are merged by question ID. Runtime metadata and original request text are omitted from this comparison.

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "Please provide copies of all emails or letters exchanged between me and my allocated social care support worker. These records should cover the period from January 2023 to the present.",
      "regime": "SAR",
      "topic": "communications with allocated social care support worker"
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I am writing to request copies of all correspondence between myself and my allocated social care support worker.",
    "This request is from <PERSON>.",
    "Replies can be sent to <EMAIL_ADDRESS>."
  ],
  "request_topics": [
    "communications with allocated social care support worker"
  ]
}
```

### R5 — Mixed personal and environmental requests with two bullet styles

- `case_id` / `group_id`: `4c1f73bf-a4df-57df-81ee-9883956f5b1f`
- Raw-text source: companion topic split line 17, `prompt[1].content.request_text`.
- Gold-label source: QuestionSlice split lines 75–85.

Keep care records and land contamination reports as separate questions, with their respective continuations. Contact placeholders and the closing thanks are ADDITIONAL in the gold labels.

**Input**

```json
{
  "body": "I need some information from Northbridge Borough Council. Please treat this as two separate requests below. I am <PERSON>. Email: <EMAIL_ADDRESS>. My postal address is <ADDRESS>. Telephone: <PHONE_NUMBER>.\n• Can you send me the minutes and notes from my most recent annual care plan review meeting and any earlier review records you hold?\nPlease include any amendments or updates made to my care or support plan in the last three years.\n* Also, could you provide me with preliminary land contamination reports you have for any sites currently under consideration for development by Northbridge Borough Council?\nI am particularly interested in reports that include details on chemicals detected during site surveys and any initial recommendations for further investigation. Thanks for your help."
}
```

**Expected extraction fields**

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "• Can you send me the minutes and notes from my most recent annual care plan review meeting and any earlier review records you hold? Please include any amendments or updates made to my care or support plan in the last three years."
    },
    {
      "question_id": "q2",
      "text": "* Also, could you provide me with preliminary land contamination reports you have for any sites currently under consideration for development by Northbridge Borough Council? I am particularly interested in reports that include details on chemicals detected during site surveys and any initial recommendations for further investigation."
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I need some information from Northbridge Borough Council.",
    "Please treat this as two separate requests below.",
    "I am <PERSON>.",
    "Email: <EMAIL_ADDRESS>.",
    "My postal address is <ADDRESS>.",
    "Telephone: <PHONE_NUMBER>.",
    "Thanks for your help."
  ]
}
```

**Expected full-endpoint fields**

Source: companion topic split line 17, `completion[0].content`. Question text and classification are merged by question ID. Runtime metadata and original request text are omitted from this comparison.

```json
{
  "questions": [
    {
      "question_id": "q1",
      "text": "• Can you send me the minutes and notes from my most recent annual care plan review meeting and any earlier review records you hold? Please include any amendments or updates made to my care or support plan in the last three years.",
      "regime": "SAR",
      "topic": "care plan review records"
    },
    {
      "question_id": "q2",
      "text": "* Also, could you provide me with preliminary land contamination reports you have for any sites currently under consideration for development by Northbridge Borough Council? I am particularly interested in reports that include details on chemicals detected during site surveys and any initial recommendations for further investigation.",
      "regime": "EIR",
      "topic": "preliminary land contamination reports for development sites"
    }
  ],
  "extraction_status": "questions_found",
  "additional_text": [
    "I need some information from Northbridge Borough Council.",
    "Please treat this as two separate requests below.",
    "I am <PERSON>.",
    "Email: <EMAIL_ADDRESS>.",
    "My postal address is <ADDRESS>.",
    "Telephone: <PHONE_NUMBER>.",
    "Thanks for your help."
  ],
  "request_topics": [
    "care plan review records",
    "preliminary land contamination reports for development sites"
  ]
}
```

## Response analysis: five cases

### S1 — Information supplied in an unavailable attachment

- Validation split line: 1.
- `family_id`: `fbacb537-5907-5b78-8950-fc683095f380`.
- Recipe: `attachment_supplied_unobservable`; form: `expansion_v4`.
- Original `input_sha256`: `b7853cc3d59ab63552dd5fe147a61241616a70a9a84c996af7f8375b03a05a1f`.
- Original `pair_sha256`: `00bfd0857598b9cb9b4801b2e079dd801abd289b4fdd710f499f91cb9f5cea3a`.

An explicit supply claim is present, but the attachment text is unavailable: supplied / unobservable, with attachment as the content source.

**Input**

```json
{
  "request": {
    "additional_text": [],
    "extraction_status": "questions_found",
    "questions": [
      {
        "question_id": "q1",
        "text": "1) Please provide a log or summary of amendments made to the force’s policing policies and guidance over the past five years.\nI am interested in records such as revision logs, change registers, or dated summaries that document the nature and timing of these amendments."
      }
    ]
  },
  "request_text": "I am writing to request information regarding changes that have been made to the Northbridge Constabulary’s policing procedures and guidance documents at the force level. I am particularly interested in understanding the amendments that have taken place over recent years.\n1) Please provide a log or summary of amendments made to the force’s policing policies and guidance over the past five years.\nI am interested in records such as revision logs, change registers, or dated summaries that document the nature and timing of these amendments.",
  "sources": [
    {
      "availability": "visible",
      "filename": null,
      "id": "email",
      "kind": "email",
      "role": "current",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nYour request asked: 1) Please provide a log or summary of amendments made to the force’s policing policies and guidance over the past five years. I am interested in records such as revision logs, change registers, or dated summaries that document the nature and timing of these amendments.\n\nWe are supplying the information requested in this question in the attached response.pdf.\n\nYours sincerely,\nInformation Rights Team"
    },
    {
      "availability": "unavailable",
      "filename": "response.pdf",
      "id": "attachment",
      "kind": "attachment",
      "role": "current",
      "sender": "authority",
      "text": null
    }
  ]
}
```

**Expected output**

```json
{
  "events": [],
  "outcomes": [
    {
      "act": "supplied",
      "answer_content_visibility": "unobservable",
      "content_source_ids": [
        "attachment"
      ],
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q1"
      }
    }
  ],
  "process_references": [],
  "process_states": null
}
```

### S2 — Delayed response with a promise to answer one question

- Validation split line: 14.
- `family_id`: `f801d6d8-7c0f-58c7-8510-c6030dde00da`.
- Recipe: `delay_with_pending`; form: `expansion_v4`.
- Original `input_sha256`: `26bc707980ae76eef8b01598026c29ac37b2a12578abc00689e50672b076c7d9`.
- Original `pair_sha256`: `08d70e1ed18bf34bf30958b44c32ada7455a118942b1df2a6b4dac95561d0798`.

Record pending and delay_notice for q1 only. The reply does not establish an outcome for q2; do not invent one or add a processing event.

**Input**

```json
{
  "request": {
    "additional_text": [],
    "extraction_status": "questions_found",
    "questions": [
      {
        "question_id": "q1",
        "text": "Could you provide a summary of the quarterly performance reports detailing how public services have met their targets over the past two years?\nSpecifically, I am interested in reports for education services, but information on other service areas is also welcome."
      },
      {
        "question_id": "q2",
        "text": "Additionally, please describe the key performance indicators that are used to monitor the delivery of public services across your administration.\nI would appreciate details on how these indicators are defined, how often they are reviewed, and any recent changes to the framework."
      }
    ]
  },
  "request_text": "I am writing to request information regarding the systems and processes used by Northbridge Public Administration to monitor and report on the performance of public services. Could you provide a summary of the quarterly performance reports detailing how public services have met their targets over the past two years? Specifically, I am interested in reports for education services, but information on other service areas is also welcome. Additionally, please describe the key performance indicators that are used to monitor the delivery of public services across your administration. I would appreciate details on how these indicators are defined, how often they are reviewed, and any recent changes to the framework.",
  "sources": [
    {
      "availability": "visible",
      "filename": null,
      "id": "email",
      "kind": "email",
      "role": "current",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nRegarding your question: Could you provide a summary of the quarterly performance reports detailing how public services have met their targets over the past two years?\nSpecifically, I am interested in reports for education services, but information on other service areas is also welcome.\n\nOur substantive response to this question is delayed. We will send that response separately.\n\nYours sincerely,\nInformation Rights Team"
    }
  ]
}
```

**Expected output**

```json
{
  "events": [
    {
      "kind": "delay_notice",
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q1"
      }
    }
  ],
  "outcomes": [
    {
      "act": "pending",
      "answer_content_visibility": "not_applicable",
      "content_source_ids": [],
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q1"
      }
    }
  ],
  "process_references": [],
  "process_states": null
}
```

### S3 — Internal review acknowledgement with an earlier refusal

- Validation split line: 9.
- `family_id`: `bb725965-bf51-5d63-9015-d616b3d7863f`.
- Recipe: `review_acknowledgement`; form: `expansion_v4`.
- Original `input_sha256`: `63643d953e0ce8f82e12fda8d7d2e101103fe350cf809cad95664676189c3862`.
- Original `pair_sha256`: `05fe1eacb7d71bb6c7df1e67c20e46e581320157c04f74fc354b3d870419a284`.

The current message acknowledges a review. The earlier refusal is context, not a new current withholding outcome.

**Input**

```json
{
  "request": {
    "additional_text": [],
    "extraction_status": "questions_found",
    "questions": [
      {
        "question_id": "q1",
        "text": "Could you provide the annual figures for the number of complaints received and the outcomes of those complaints for each year from 2020 to 2023?\nI am interested in data broken down by year, including the number of complaints received, the number acknowledged within target times, the number resolved in favour of the complainant, the number not upheld, and any other standard outcome categories you record."
      }
    ]
  },
  "request_text": "I am writing to ask about the annual statistics relating to complaints made to and handled by Northbridge Public Administration in recent years. Could you provide the annual figures for the number of complaints received and the outcomes of those complaints for each year from 2020 to 2023? I am interested in data broken down by year, including the number of complaints received, the number acknowledged within target times, the number resolved in favour of the complainant, the number not upheld, and any other standard outcome categories you record.",
  "sources": [
    {
      "availability": "visible",
      "filename": null,
      "id": "prior_decision",
      "kind": "email",
      "role": "prior",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nIn our earlier response, we refused the information requested.\n\nYours sincerely,\nInformation Rights Team"
    },
    {
      "availability": "visible",
      "filename": null,
      "id": "email",
      "kind": "email",
      "role": "current",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nYour request asked: Could you provide the annual figures for the number of complaints received and the outcomes of those complaints for each year from 2020 to 2023? I am interested in data broken down by year, including the number of complaints received, the number acknowledged within target times, the number resolved in favour of the complainant, the number not upheld, and any other standard outcome categories you record.\n\nWe acknowledge your request for an internal review. The review has not yet been decided.\n\nYours sincerely,\nInformation Rights Team"
    }
  ]
}
```

**Expected output**

```json
{
  "events": [
    {
      "kind": "internal_review_acknowledgement",
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q1"
      }
    }
  ],
  "outcomes": [],
  "process_references": [],
  "process_states": null
}
```

### S4 — Partly revised internal review: supplied and withheld

- Validation split line: 3.
- `family_id`: `f53158ee-95c8-582d-911e-b2d5406e8223`.
- Recipe: `review_partly_revised`; form: `expansion_v4`.
- Original `input_sha256`: `d44f3b8e740c96c29d74eab42a5f1bba286a7d4e6297a46d20b9fa6aa15f342e`.
- Original `pair_sha256`: `01a7b0f8c54591b699f449d755703a3cc5d43ffe13df7eb3c7b346df16de2f37`.

The current review releases q1 and maintains withholding for q2 under FOIA section 36. Record the whole-request partly_revised review decision and both question outcomes.

**Input**

```json
{
  "request": {
    "additional_text": [],
    "extraction_status": "questions_found",
    "questions": [
      {
        "question_id": "q1",
        "text": "(a) Please provide the population growth rates and demographic projections currently used by the Council when assessing the need for new school places over the next five years."
      },
      {
        "question_id": "q2",
        "text": "1. Please detail the assumptions regarding sibling priority and any other priority admission categories that are factored into school place planning and related resource allocation decisions."
      }
    ]
  },
  "request_text": "Northbridge Borough Council is responsible for planning and\nallocating school places across the borough. This planning\ntypically relies on a set of underlying demographic and\nenrolment assumptions that inform decisions about when and\nwhere to create new school capacity.\n(a) Please provide the population growth rates and demographic projections currently used by the Council when assessing the need for new school places over the next five years.\nIn addition to overall population trends, planning documents\nmay also incorporate assumptions about enrolment patterns,\nincluding priority admissions and sibling policies.\n1. Please detail the assumptions regarding sibling priority and any other priority admission categories that are factored into school place planning and related resource allocation decisions.",
  "sources": [
    {
      "availability": "visible",
      "filename": null,
      "id": "prior_decision",
      "kind": "email",
      "role": "prior",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nIn our earlier response, we refused the information requested.\n\nYours sincerely,\nInformation Rights Team"
    },
    {
      "availability": "visible",
      "filename": null,
      "id": "email",
      "kind": "email",
      "role": "current",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nOur internal review is complete. We have partly revised our earlier decision: we now release the information for the first question, but maintain the refusal for the second question.\n\nRegarding your question: (a) Please provide the population growth rates and demographic projections currently used by the Council when assessing the need for new school places over the next five years.\n\nWe are supplying the information for the first question below:\n\nThe Council currently uses annual population growth rates of approximately 1.2% for the local authority area, based on ONS mid-year estimates and local authority projections. For demographic projections over the next five years, the Council employs a cohort-component model that factors in birth rates, migration patterns, and household formation. Recent projections suggest a steady increase in the 0-15 age cohort, with an anticipated rise of 8.5% by 2029. These figures inform the annual review of primary and secondary school capacity needs, which includes rolling five-year forecasts updated quarterly. The methodology aligns with the Department for Education’s guidance on school place planning.\n\nRegarding your question: 1. Please detail the assumptions regarding sibling priority and any other priority admission categories that are factored into school place planning and related resource allocation decisions.\n\nFor the second question, we continue to withhold the information requested under section 36 of the Freedom of Information Act 2000.\n\nYours sincerely,\nInformation Rights Team"
    }
  ]
}
```

**Expected output**

```json
{
  "events": [
    {
      "kind": "internal_review_decision",
      "references": [],
      "review_result": "partly_revised",
      "scope": {
        "kind": "whole_request"
      }
    }
  ],
  "outcomes": [
    {
      "act": "supplied",
      "answer_content_visibility": "visible",
      "content_source_ids": [
        "email"
      ],
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q1"
      }
    },
    {
      "act": "withheld",
      "answer_content_visibility": "not_applicable",
      "content_source_ids": [],
      "references": [
        {
          "instrument": "foia_2000",
          "provision": "36",
          "role": "basis_for_withholding",
          "subsection": null
        }
      ],
      "scope": {
        "kind": "question",
        "question_id": "q2"
      }
    }
  ],
  "process_references": [],
  "process_states": null
}
```

### S5 — Visible attachment with supplied, not-held and pending outcomes

- Validation split line: 431.
- `family_id`: `1c91cf1d-b6dc-5cb7-bbee-c8239c16317d`.
- Recipe: `attachment_three_numbered_supplied_not_held_pending`; form: `targeted_v4`.
- Original `input_sha256`: `1955632d3dcd5e8f14613bf5724fe8fa63c0e77c1ebd5c8b827f773b166247dd`.
- Original `pair_sha256`: `0bd7220907974cc657ae4a429ed311fb97e1921f93655a304d7d7d9a6a2923c7`.

Resolve the attachment’s out-of-order decisions to q3, q1 and q2. The answer content is in attachment, not the covering email.

**Input**

```json
{
  "request": {
    "additional_text": [],
    "extraction_status": "questions_found",
    "questions": [
      {
        "question_id": "q1",
        "text": "Provide a summary of the mandatory training completed by\nofficers within the last three years.\nThis request covers all ranks of sworn officers and includes\nboth classroom-based and online modules."
      },
      {
        "question_id": "q2",
        "text": "Disclose details of advanced training courses attended by\nofficers in the last five years, including the dates,\nproviders, and accreditation levels."
      },
      {
        "question_id": "q3",
        "text": "Supply an overview of the current approach to mandatory\ntraining delivery within the force, including frequency and\nevaluation methods."
      }
    ]
  },
  "request_text": "I am writing to request information regarding officer\ntraining records held by your authority. I am seeking\nfactual summaries of the training undertaken by officers,\nspecifically as recorded in your official files.\nProvide a summary of the mandatory training completed by\nofficers within the last three years.\nThis request covers all ranks of sworn officers and includes\nboth classroom-based and online modules.\nDisclose details of advanced training courses attended by\nofficers in the last five years, including the dates,\nproviders, and accreditation levels.\nSupply an overview of the current approach to mandatory\ntraining delivery within the force, including frequency and\nevaluation methods.",
  "sources": [
    {
      "availability": "visible",
      "filename": null,
      "id": "email",
      "kind": "email",
      "role": "current",
      "sender": "authority",
      "text": "Dear Sir or Madam,\n\nOur reply is attached.\n\nYours sincerely,\nInformation Rights Team"
    },
    {
      "availability": "visible",
      "filename": "response.pdf",
      "id": "attachment",
      "kind": "attachment",
      "role": "current",
      "sender": "authority",
      "text": "Decision 1: 3rd question.\n\nWe will send a substantive response concerning the 3rd question separately.\n\nDecision 2: 1st question.\n\nWe are supplying the information for the 1st question below:\n\nBetween 2021 and 2024, all sworn officers completed a series of mandatory training modules. Core topics included public order management, use of force, mental health awareness, safeguarding, and equality and diversity. Classroom-based sessions were primarily delivered annually, while online refresher courses were available quarterly. Training was provided by the Force Training Academy and accredited external providers such as the National Police College and the College of Policing. Evaluation was conducted through post-course assessments and periodic practical scenario exercises, with attendance and performance records maintained centrally.\n\nDecision 3: 2nd question.\n\nFollowing a search, we do not hold the information for the 2nd question."
    }
  ]
}
```

**Expected output**

```json
{
  "events": [],
  "outcomes": [
    {
      "act": "pending",
      "answer_content_visibility": "not_applicable",
      "content_source_ids": [],
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q3"
      }
    },
    {
      "act": "supplied",
      "answer_content_visibility": "visible",
      "content_source_ids": [
        "attachment"
      ],
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q1"
      }
    },
    {
      "act": "not_held",
      "answer_content_visibility": "not_applicable",
      "content_source_ids": [],
      "references": [],
      "scope": {
        "kind": "question",
        "question_id": "q2"
      }
    }
  ],
  "process_references": [],
  "process_states": null
}
```

## Source-file checksums

SHA-256 of the downloaded validation files, before selecting examples:

- QuestionSlice validation_heldout_topic.jsonl: `791c37fa7a9c5f8154ed1cfd03707669009ee3550702bb86defa4d4879e80022`
- Grounded topic validation_heldout_topic.jsonl: `8c5f5a4f12c563761d8bfa99bb476b7fb92051ca9ae3dad2573a6c94f1ee798a`
- Response v3 validation.jsonl: `5621072fc1de81f4880c6cd0f5e9b46e7e0ecb7390956ad356c7c35b93a6d345`

## Reviewed application alternatives

The JSON blocks above remain the original dataset expectations. The application
also accepts explicit reviewed alternatives stored separately in
`src/llm_management/foi/example_data/validation.json`. These were observed on both
CPU and Exoscale in the paid validation run of 12 October 2026 and approved as
equivalent in this repository:

- R1–R5: the observed complete topic-classification outputs, preserving all question
  IDs, order and FOI/EIR/SAR regimes, with alternative topic wording.
- R3: the two qualification sentences attached to q1 instead of additional_text.
  All content is retained and the same single question is identified.

The tests continue to report exact gold fidelity separately. Other topic wording,
missing content, changed regimes, IDs or extra questions are not automatically
accepted. Mocks continue to return original dataset outputs.
