# Laby Agent — VPS deployment runbook

## Production identity

| Item | Value |
|------|-------|
| Public name | `https://ai.dentnode.com` |
| Runtime | DentNode VPS, Docker, Caddy |
| Deployment | `.github/workflows/deploy-vps.yaml` |
| Image | `ghcr.io/ramancodians/ai.dentnode.com@sha256:<digest>` |
| Health | `GET /health` |

Cloud Run hosting was retired on 2026-09-19. There is no Cloud Run deployment
workflow or Cloud Run rollback path.

## Normal deployment

A push to `main` runs the VPS workflow. It checks out the exact commit, runs the
test suite, builds and pushes an immutable GHCR image, and sends only its digest
to the service-scoped restricted dispatcher.

The dispatcher must:

1. pull by digest using the short-lived GitHub token;
2. start and health-check a private candidate;
3. verify the candidate from the Caddy network;
4. atomically switch the public route; and
5. retain the previous digest for rollback.

See `deploy/vps/README.md` for the complete host contract and required GitHub
environment secrets.

## Health checks

Public liveness:

```bash
curl -fsS https://ai.dentnode.com/health
```

Expected response fields include `status=healthy`, `service=laby-adk`, and the
configured model provider. This endpoint does not make a billable model call.

From the VPS Caddy network, verify a candidate before promotion:

```bash
docker exec dentnode-caddy wget -qO- http://dentnode-ai-candidate:8080/health
```

Then call `GET /text-to-speech/voices` with the internal key to exercise service
authentication without invoking a model.

## Rollback

Use the restricted dispatcher to restore the previously retained immutable
digest. Do not rebuild an old commit, deploy a mutable tag, or recreate the
former Cloud Run service.

After rollback, verify both:

```bash
curl -fsS https://ai.dentnode.com/health
curl -fsS https://app.dentnode.com/api/health/ready
```

## Configuration and secrets

Runtime configuration is held in the root-owned VPS files under
`/opt/dentnode/apps/ai.dentnode.com/env/`:

- `runtime.env` — non-secret runtime configuration
- `release.env` — immutable release metadata
- `secrets.env` — mode `0600`, root-owned secrets

The service must remain attached only to `dentnode-edge` and
`dentnode-telemetry`; it must not join `dentnode-data`. The Node application
calls the stable `https://ai.dentnode.com` URL using the shared internal key.

## Observability

Traces are exported to the private host collector at
`http://dentnode-telemetry-agent:4318/v1/traces`. No OTLP port is public. Use
SigNoz for service health, errors, and trace investigation.
