# D10 call recording analysis

`POST /d10/call-audio/analyze` accepts a recording uploaded by the D10 backend.
It never fetches a URL or looks up a patient. D10 must authorize clinic ownership,
obtain a READY recording through the shared Calling Service, and download it
before calling this endpoint. Browser/mobile callers do not supply recording URLs.

Authenticate with `x-d10-internal-key`. Authentication and the overall body cap
run before multipart parsing. The app's `x-internal-key` does not authorize this
route. The existing app-only `/internal/call-audio/analyze` contract is unchanged.

Send multipart fields:

| Field | Value |
| --- | --- |
| `file` | MP3 bytes, `audio/mpeg` or `audio/mp3`, at most 25 MiB (or a lower configured audio cap) |
| `call_id` | Authorized shared service call ID |
| `analysis_id` | D10's immutable analysis job ID |
| `recording_id` | Canonical recording ID obtained server-side |
| `recording_fingerprint` | Lowercase SHA-256 of D10's canonical call/recording identity and READY metadata |
| `usage_context` | JSON object described below |

The usage context contains only `clinic_id`, `user_id`, `actor_id`,
`conversation_id`, `source_message_id`, and `correlation_id`. The actor equals the
authorized user; conversation is `call:<call_id>`; source message equals
`analysis_id`; correlation equals the shared `call_id` so existing call-cost
views can attribute this spend. No app lab ID, reservation ID, or additional
patient facts are accepted. D10 remains responsible for authorization and cache.

After MIME, byte signature, size and identity validation, the existing durable
execution claim binds clinic, call, recording and fingerprint. A second request
for that artifact returns HTTP 409 even with a new analysis job. A lost response
or uncertain provider failure must be reconciled; it does not authorize a paid
retry. Validation failures occur before the claim and any model request.

Success returns:

```json
{"transcript":"Recorded words.","summary":"Concise recorded summary.","summary_error":null,"summary_source":"model","summary_model_skipped":false}
```

If transcription succeeds but summarization fails, HTTP 200 preserves the
transcript, with `summary: null` and `summary_error: "SUMMARY_UNAVAILABLE"`.
Transcription failure returns sanitized HTTP 502 and retains the artifact claim.
There are no automatic model retries. The AI service stores no audio/transcript;
D10 owns the immutable job/cache and displays any partial-result limitation.

Wholly greeting/acknowledgement-only transcripts use the deterministic summary
`Greeting only; no clinical details recorded.` with
`summary_source: "deterministic_greeting"` and `summary_model_skipped: true`.
This D10-only mode makes no summary model request and emits no summary usage
event. Other call summaries use system instructions forbidding invented replies,
turns, roles or clinical details. Diarization IDs do not establish speaker roles.
The recorded transcript remains verbatim. Previously billed raw summaries and
receipts remain audit records even if D10 corrects their greeting-only display;
such a display correction does not retroactively mark that model request skipped.

The route reuses the existing OpenRouter transcription and summary models.
Completed transcription and summary stages use the existing `notes_transcription`
and `notes_summary` D10 usage features; they never report to the app ledger.
Observed provider request IDs, usage and cost are retained in the durable D10
outbox. Missing receipts or costs remain unknown. Failed attempts are recorded
with unknown cost rather than an invented zero charge. A summary failure still
records the completed transcription and the failed summary attempt separately.

The local E2E runtime uses a dedicated outbox and local D10 callback. This
endpoint does not start a voice worker, initiate calls, deploy code or apply a
database migration. The existing URL-based `/audio-to-text` response is unchanged.
