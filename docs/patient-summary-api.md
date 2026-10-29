# D10 patient directory summary

`POST /d10/patient-summary` is a D10-only, stateless text-facts endpoint. It uses
the existing OpenRouter `LABY_MODEL` adapter (DeepSeek by default), with no tool
loop, identity search, database read, audio download or external source fetch.
It returns a short recorded clinical problem, not a new diagnosis or an action.

Authenticate with `x-d10-internal-key`. The app service's `x-internal-key` does
not authorize this endpoint. D10 must authorize the clinic and patient, collect
the source facts, and create an immutable dispatch snapshot before requesting it.
Do not send patient names, contact details or identifiers as separate model data.

```json
{
  "version": 1,
  "patient_id": "patient_opaque",
  "facts": [
    {"source_id": "note_opaque", "kind": "note", "text": "Patient reports tooth pain."}
  ],
  "usage_context": {
    "clinic_id": "clinic_opaque",
    "user_id": "owner_at_dispatch",
    "job_id": "job_opaque",
    "summary_id": "job_opaque",
    "reservation_id": "reservation_opaque"
  }
}
```

Facts are supplied newest first. Supported kinds are `note`, `diagnosis`,
`treatment`, `appointment` and `call_summary`. A treatment-only fact cannot support
the returned condition: a procedure name is not a recorded complaint/diagnosis.
There may be at most 20 facts, each at most 2000 characters, totaling at most 12000.
Source IDs must be unique. IDs are opaque ASCII letters/numbers/underscore,
hyphen or colon, at most 200 characters. `job_id` and `summary_id` must match.

```json
{
  "version": 1,
  "clinical_label": "Tooth pain",
  "evidence": [{"source_id": "note_opaque", "quote": "Patient reports tooth pain."}]
}
```

A supported label has exactly two Unicode letter/mark words, at most 60 characters,
with no numbers or control characters. Exactly one evidence quote, at most 240
characters, must be copied verbatim from the referenced fact. The label's words
must both appear in that quote; reordering and dropping function words are
allowed (for example, “pain in my tooth” → “Tooth pain”). Unsupported translations
and newly invented medical terms are rejected. Negated/resolved complaints, administrative
content, family-history ambiguity, question-only evidence, instruction-like text,
test-call/hypothetical/example speech, unknown source IDs and malformed model
output abstain conservatively:

```json
{"version": 1, "clinical_label": null, "evidence": []}
```

This conservative extractive validation can return null for a legitimate problem
that cannot be represented by two grounded words. The label is a directory aid,
not proof of a current diagnosis, attendance or patient identity. D10 keeps the
evidence/cache and presents an honest unavailable state where needed. There is
no extra narrative-summary field or model retry to repair unsupported output.

## Execution and usage

Empty facts return null without a model call or execution claim. Otherwise a
durable pre-provider claim uses `patient-summary:<clinic_id>:<summary_id>`. Any
duplicate returns HTTP 409 without a cached patient response or a second model
request. D10 owns the immutable input fingerprint and job/result cache; a changed
body with the same job identity cannot authorize another model execution.

One model request has a 20-second timeout. Instructions are in the system message;
the bounded facts are separate delimited user data. Prompts, facts, output and
provider exception bodies are not logged. An upstream/usage-persistence failure
returns a sanitized HTTP 502. Malformed/unsupported output after an upstream
success returns null and still records that successful, billable request.

Usage is durably enqueued in the existing D10 usage outbox, including failures:

| Field | Value |
| --- | --- |
| `feature` | `patient_clinical_summary` |
| `event_type` | `ai.model_call` |
| `actor_id` | `system:patient-summary` |
| `model_call_index`, `attempt` | `1`, `1` |
| `event_id` | Existing builder UUID, retained by the outbox |
| `idempotency_key` | `patient-summary:<clinic_id>:<summary_id>` |
| `correlation_id` | `patient-summary:<summary_id>` |
| `causation_id`, `job_id` | Supplied job ID |
| `reservation_id` | Supplied D10 reservation |
| `user_id` | Owner from D10's dispatch snapshot |
| `status` | `ok` for an upstream result, including invalid label output; `error` on failure |

Observed model, request ID, tokens, provider cost and latency retain the existing
metering format. Missing cost remains unknown; no zero charge or savings claim is
invented. Existing model-call pricing applies. A lost HTTP response or occupied
claim must be reconciled against D10's job/cache and usage; it must not trigger an
automatic new paid request or release an ambiguous hold.

## Existing audio consumers

`/audio-to-text` and `/internal/call-audio/analyze` are unchanged, including
their summary response shape and separate transcription/summary metering.
No patient-label flag was added to either audio endpoint. App consumers retain
their existing behavior. D10 patient labels use this dedicated endpoint and
persisted authorized text facts, rather than regenerating on a directory GET.

No migration, model deployment or live patient-data test is part of this change.
