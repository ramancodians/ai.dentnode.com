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
