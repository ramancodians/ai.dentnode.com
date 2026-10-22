# Shared reception tools (prototype)

The AI service exposes a transport-independent gateway for voice/chat services.
It does not own patient data or place calls. D10 owns clinic operations; app owns
lab cases and staff operations. There is no automatic clinic-to-lab identity mapping.

## Contract

`POST /reception/tools/catalog` accepts a scope:

```json
{"platform":"app","context":{"lab_id":"authorized-lab","user_id":"active-admin","correlation_id":"trace-id"}}
```

Use server-only `x-internal-key` for app and `x-d10-internal-key` for D10.
The response contains OpenAI function declarations in `tools`.

`POST /reception/tools/execute`:

```json
{
  "scope": {"platform":"app","context":{"lab_id":"authorized-lab","user_id":"active-admin","correlation_id":"trace-id"}},
  "name":"find_case",
  "parameters":{"query":"case-123","limit":5},
  "tool_call_id":"stable-invocation-id"
}
```

For D10, scope is `{"platform":"d10","context":{...}}`, with the complete
existing D10RequestContext: clinic_id, user_id, conversation_id, actor_id,
actor_role, timezone, source_message_id, correlation_id. Optional causation_id
and reservation_id remain supported. Obtain it from authorized platform ingress
and persisted contact/conversation/message records. A caller number is not proof
of patient identity. The model supplies only name/parameters, never scope.

Successful execution returns `{success,platform,tool,tool_call_id,result}`.
Interpret the domain result: a confirmation request, empty match, or draft is
not a completed booking. Existing D10 later-message confirmation remains required.
No execution request is automatically retried. A 502/network timeout may have
an unknown outcome: reconcile with the owning backend before retrying. A
tool_call_id is correlation, not a universal idempotency guarantee.

The voice worker can use `agent.reception.router.catalog(scope)` and
`execute(Execution(...))` in process. These functions assume the orchestrator
has already authenticated and constructed the trusted scope. HTTP callers must
authenticate separately for each platform. Do not expose internal keys to clients.
Catalog and execution make no model calls. Optional selection below invokes Jev
and meters that decision separately.

## Optional Jev tool selection

`POST /reception/tools/select` accepts `{ "scope": <same trusted scope>,
"query": "I want to book an appointment" }` with the same platform key.
It returns function declarations in `tools` plus advisory `selection` metadata.
Use those declarations for the next conversation-agent turn; fall back to
`/catalog` whenever the shortlist cannot complete the request. Existing agents
are not automatically switched to this endpoint.

Jev chooses among bounded task families, not patients or tool arguments. Only the
short query (maximum 256 characters) and generic family labels go to the model;
identity context, catalog schemas and patient records are not sent. Keep queries
minimal. The model deadline is two seconds; catalog, authorization and metering
add latency beyond that. A single eligible family skips Jev. Ambiguity, invalid
selection or Jev failure returns the full catalog. Supporting and unmapped tools
are retained. This does not execute anything or replace backend confirmation.

Each attempt is metered through app AI usage or the D10 durable usage outbox
(the AI server's existing delivery loop flushes it). Response metadata includes
catalog/selected counts, decision latency and known cost; no savings are claimed
until real workloads are measured. DeepSeek conversation fallback remains the
caller's responsibility; selection never silently invokes a second model.

## Available operations

| Owner | Tools |
| --- | --- |
| D10 | Existing slot search, booking/rescheduling/cancellation, approved clinic knowledge, authorized CRM/history, staff handoff and existing confirmed calling tools |
| D10 additions | get_reception_status, request_callback, list_reception_work_items, get_patient_follow_up_status |
| App | find_case, staff_list, task_detail, prepare_follow_up, call_history, call_summary |

Always fetch the catalog for the current scope instead of assuming every tool
is available. D10 has role-dependent tools and backend execution checks.
App reception tools require active lab administrator membership and retain
existing calling permission checks. They are staff operations, not a public
patient lookup API. `prepare_follow_up` only prepares a task preview; create it
through the existing Tasks UI. It cannot send, schedule, or confirm a reminder.
App uses `/api/internal/laby-tools/reception/catalog` and `/.../{tool}` internally,
with context, parameters and tool_call_id in the same trusted envelope (catalog
accepts context only).

## Customer requirements and ownership

| Requirement | Tool support / remaining service work |
| --- | --- |
| Missed calls | D10 durable callback queue; telephony ingress and automatic dialling owned by calling service |
| After hours | Clinic hours/holidays + callback deadline; voice routing/24-hour availability owned by calling service |
| Appointment booking | Existing D10 availability + confirmation-gated booking |
| Appointment enquiries | Approved clinic knowledge; missing facts require handoff |
| Follow-ups | D10 follow-up status; app staff task preview. Scheduling/dispatch stays in existing backend workflows |
| Cancellation/no-show | Existing D10 cancellation and confirmation history; automated call dispatch not added |
| Call overload | Voice infrastructure concurrency, not a business tool |
| Language | Voice/STT/TTS orchestration, not implemented here |
| Lead leakage | Callback capture; lead scoring/campaign automation not added |
| Emergency | Existing human handoff; no new clinical triage or guaranteed emergency response |
| Consistent answers | Use approved knowledge, abstain on missing facts |
| Tracking | Existing D10 activity and app history/transcripts; conversion attribution not added |
| Patient history | D10 authorized own-patient/staff tools; app lab case search remains separate |
| Doctor availability | Existing D10 slot search |
| Payment/insurance | Approved clinic knowledge only; no eligibility decisions |
| Multichannel | Reusable gateway; voice ingress must establish authorized persisted D10 context |

## Configuration and rollout

Uses existing D10_INTERNAL_BASE_URL / D10_INTERNAL_KEY and
NODE_INTERNAL_BASE_URL / INTERNAL_API_KEY. Deploy backend additions before AI
consumers, through the existing VPS GitHub workflows. No schema migration.
Prototype is local/uncommitted until reviewed and deployed. Live voice wiring
and end-to-end calls are a separate integration step owned by the calling agent.
