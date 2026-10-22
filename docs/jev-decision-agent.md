# Shared Jev decision agent

`agent/jev.py` exposes one stateless `JevDecisionAgent` for typed, non-generative decisions. It calls OpenRouter's [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request) with `model: "typesafe/jev-1.13"`, `state`, and `questions`. The [TypeSafe SDK guide](https://openrouter.ai/docs/guides/community/typesafe-sdk) documents the same System One request shape through its SDK. Jev returns `choice`, `score`, or yes/no `noul` answers, not free-form text.

Call `jev_agent.decide` when a workflow needs a bounded classification or choice. `jev_agent.match_text` combines a candidate choice with an absolute match question and can abstain. Both return usage, cost, latency, model, and request ID so the caller can meter every attempt. Use Jev first for these small semantic decisions; use DeepSeek when Jev cannot make the decision or the task requires generated text or tool-calling conversation. Keep exact IDs, dates, numbers, authorization, and tenant filtering in code.

## Search integration

The app and D10 backends perform tenant-scoped database search and return candidates. `agent/jev_search.py` sends only short candidate labels to Jev and adds an advisory `jev_match` index without deleting or reordering results. Laby uses it for `find_case`, `find_doctor`, and queried `staff_list`. D10 uses it for `search_patients`, `prepare_patient_call`, `search_medicines`, `search_clinic_knowledge`, and `search_call_conversations`.

Exact identifiers and single results bypass semantic matching. Lists over 20 candidates and queries over 200 characters keep the original backend result without a Jev suggestion. Patient identity remains ambiguous until resolved by the user; a Jev suggestion never authorizes a clinical answer, booking, or call. Laby reports Jev usage to the Node AI ledger, while D10 records it in the usage outbox before returning the tool result.

Only the fields needed for matching should leave this service. Short free-text labels may still contain patient data. OpenRouter [states it does not offer a HIPAA BAA](https://openrouter.ai/blog/insights/governing-team-ai-spend/); workloads requiring one need another processing path. Keep prompts and candidate text out of logs, preserve internal-key boundaries, and measure actual cost and latency before claiming savings.
