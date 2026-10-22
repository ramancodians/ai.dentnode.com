# VPS deployment contract

This directory is the reviewed source template for the root-owned deployment at
`/opt/dentnode/apps/ai.dentnode.com`. The GitHub workflow does not copy these
files or execute arbitrary shell commands. It sends one immutable GHCR digest to
the VPS service-scoped dispatcher.

`deploy-vps.yaml` is the only automatic production deployment: every push to
`main` runs the full test suite, builds the exact tested commit, publishes it to
GHCR by immutable digest, and sends that digest to the restricted dispatcher.
Pull requests run the same test suite in `ci.yaml` without registry, VPS, or GCP
credentials. The retired Cloud Run workflow is no longer present, and this
repository has no staging deployment path.

## Server prerequisites

- Create `/var/lib/dentnode/ai-outbox` as `10001:10001` with mode `0700`.
- Copy `compose.yaml` and the environment templates into the service directory.
- Create `env/runtime.env`, `env/release.env`, and `env/secrets.env` as root-owned
  regular files. Secrets must have mode `0600`.
- Attach the service only to `dentnode-edge` and `dentnode-telemetry`. It does
  not use MySQL and must not join `dentnode-data`.
- Support `deploy-auth` in the restricted dispatcher. The workflow streams its
  short-lived `GITHUB_TOKEN` over SSH stdin for the candidate pull; the server
  must log in only for that pull and remove the temporary Docker auth directory.
- Add a repository-specific SSH public key using the existing restricted
  `dentnode-deploy-ssh` forced command.
- Extend the root dispatcher with only this service and image pattern:
  `ghcr.io/ramancodians/ai.dentnode.com@sha256:<64 lowercase hex>`.

The repository's GitHub `production` environment needs these secret names:

- `VPS_HOST`
- `VPS_PORT`
- `VPS_DEPLOY_USER`
- `VPS_KNOWN_HOSTS`
- `VPS_SSH_PRIVATE_KEY`

`VPS_KNOWN_HOSTS` is pinned out of band. The workflow never runs
`ssh-keyscan`, accepts a mutable image tag, or grants an interactive SSH shell.

The controller must serialize deployments, validate all required secret names,
pull by digest, start and health-check a private candidate, test it from the
Caddy network, atomically update the route, and retain the previous digest for
rollback. It must never restart or reconfigure app.dentnode.com.

## Private-first checks

Before installing the public Caddy site, check the candidate from the edge
network:

```sh
docker exec dentnode-caddy wget -qO- http://dentnode-ai-candidate:8080/health
```

Then call `GET /text-to-speech/voices` with the internal key. It exercises
authentication without making a billable model request. Only after both checks
pass should the controller render `caddy-site.template`, validate Caddy, reload
it, and verify `https://ai.dentnode.com/health`.

The current `/health` endpoint proves that configuration validation and the
SQLite outbox initialization completed. It deliberately does not make live
OpenRouter, app.dentnode.com, or D10 calls.

## Optional inbound voice worker

The image entrypoint is `python -m runtime`. By default
`VOICE_WORKER_ENABLED=false` directly execs the existing uvicorn API, with no
voice process. Only enable it after installing the same LiveKit project
credentials used by Calling Service into the root-owned secret file:
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`.

Set `LIVEKIT_AGENT_NAME=dentnode-receptionist`,
`CALLING_SERVICE_URL=https://calling.dentnode.com`, and
`VOICE_WORKER_IDLE_PROCESSES=1` (allowed 0–2) in runtime configuration.
`VOICE_WORKER_ENABLED=true` launches the API and a separate
`python -m livekit.agents start voice_call/worker.py` child. The named worker
accepts explicit dispatches only. It binds SDK health to loopback `8081`, never
the external API port. Provision memory for both processes; 2 GiB is the
initial reviewed VPS allocation, subject to measured usage.

Container health uses `python -m runtime --health` (equivalently
`python runtime.py --health`), requiring API health, a live worker PID, the
public SDK `worker_registered` acknowledgement, and SDK health. The enabled
worker uses `max_retry=0`: a connection loss makes SDK health fail immediately
instead of appearing healthy through sixteen reconnect attempts. The
supervisor gives startup 90 seconds and exits nonzero after repeated health
failure or either child exiting, so Compose restarts the container. This
couples API and worker availability; split them into separate services if
voice volume warrants independent scaling.

Registration is not an end-to-end audio, model, recording, or patient-call
check. The API `/health` endpoint alone does not prove worker readiness.
Promotion must include the composite container health check, and Calling
Service's AI routing must remain gated until registration is verified.

On SIGTERM/SIGINT the supervisor stops the worker parent first, allowing the
SDK to drain calls (310 seconds, covering the 300-second call cap, plus
30-second process/session shutdown) while the API
remains available for metering. Remaining worker descendants are then killed,
followed by API shutdown, within a 350-second budget. Keep Compose's
`stop_grace_period: 360s`. Never send the first termination signal to the whole
worker process group, which would kill active jobs before they can drain.

Inbound audio input/output stays disabled until Calling Service acknowledges
the worker's started callback. A rejected claim or absent caller closes the
job without a greeting. Final transcripts are sent after metering with bounded
callback retries; no durable transcript callback outbox exists yet, so a
prolonged callback outage may lose that transcript. Service-side terminal
reconciliation remains required.

## App-to-AI networking

Initially set the Node application's `LABY_AGENT_URL` to
`https://ai.dentnode.com`. Do not point it at a blue/green candidate name, which
changes at promotion. A stable private routing alias can replace the HTTPS URL
later without changing this service.

## Application tracing

The service exports sampled traces to the host OpenTelemetry collector at
`http://dentnode-telemetry-agent:4318/v1/traces` over the private
`dentnode-telemetry` Docker network. No OTLP port is published on the VPS.
Before deploying this revision, the collector must expose an OTLP/HTTP receiver
on `0.0.0.0:4318` and include that receiver in its traces pipeline to SigNoz.

`OTEL_TRACES_SAMPLER_ARG` controls root-trace head sampling and defaults to `0.10`;
upstream parent decisions are honored. The immutable image digest is exported
as `service.version`, so SigNoz can separate releases.

Telemetry has an exporter-boundary privacy allowlist. Inbound FastAPI spans
contain method, registered route template, response status, protocol and
duration. Outbound HTTPX spans contain method, destination hostname/port,
response status and duration. Request/response bodies, headers, query strings,
concrete route parameters, client addresses, exception messages/events,
baggage, arbitrary attributes and secrets are never exported. Do not replace
this with broad OpenTelemetry auto-instrumentation.
