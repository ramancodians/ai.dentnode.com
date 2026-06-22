# Laby Agent — Deployment Runbook

## Service identity

| Item | Value |
|------|-------|
| GCP project | `app-dentnode-com` |
| Region | `asia-south2` |
| Cloud Run service | `laby-agent` |
| Artifact Registry repo | `asia-south2-docker.pkg.dev/app-dentnode-com/dn-dashboard` |
| Ingress | Internal only (no public internet exposure) |

## Secrets (GCP Secret Manager)

| Secret Manager name | Injected env var |
|---------------------|-----------------|
| `LABY_INTERNAL_API_KEY` | `INTERNAL_API_KEY` |
| `LABY_DEEPSEEK_API_KEY` | `DEEPSEEK_API_KEY` |

To rotate a secret:
```bash
echo -n "new-value" | gcloud secrets versions add LABY_INTERNAL_API_KEY \
  --data-file=- --project=app-dentnode-com
# Then redeploy via GitHub Actions (push to main or workflow_dispatch).
```

## Normal deployment

Push to `main` — the GitHub Actions workflow (`cloud-run-deploy.yaml`) handles
everything: run tests → build → canary at 10% → health check → promote to 100%.

To trigger manually without a code change:
```
GitHub → Actions → "Deploy Laby Agent to Cloud Run" → Run workflow
```

## Manual rollback

If a bad deploy gets past the canary:

```bash
# List the two most recent revisions
gcloud run revisions list \
  --service=laby-agent --region=asia-south2 --project=app-dentnode-com \
  --sort-by="~DEPLOYED" --limit=3 --format="table(name,status.conditions[0].type)"

# Route 100% to the known-good revision
gcloud run services update-traffic laby-agent \
  --region=asia-south2 --project=app-dentnode-com \
  --to-revisions=laby-agent-XXXXXXXX=100
```

## Health check

```bash
# Get the service URL
URL=$(gcloud run services describe laby-agent \
  --region=asia-south2 --project=app-dentnode-com --format="value(status.url)")

# The service is internal-only; call from a VM or Cloud Shell inside the VPC
curl -sf "${URL}/health" | jq .
# Expected: {"status":"healthy","service":"laby-adk","model":"deepseek/deepseek-chat","provider":"deepseek"}
```

## Smoke test (from inside the VPC)

```bash
curl -sf -X POST "${URL}/agent/run" \
  -H "Content-Type: application/json" \
  -H "x-internal-key: ${INTERNAL_API_KEY}" \
  -d '{"lab_id":"<real-lab-id>","user_id":"test","question":"how many cases today?"}' \
  --no-buffer
```

Expect a stream of NDJSON lines ending in `{"type":"done"}`.

## Logs

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="laby-agent"' \
  --project=app-dentnode-com --limit=100 --format=json | jq '.[].jsonPayload'
```

Filter by severity:
```bash
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="laby-agent" severity>=ERROR' \
  --project=app-dentnode-com --limit=50
```

## Common failure modes

### Service returns 500 on startup

Check that both secrets are present and the latest version is accessible:
```bash
gcloud secrets versions list LABY_INTERNAL_API_KEY --project=app-dentnode-com
gcloud secrets versions list LABY_DEEPSEEK_API_KEY --project=app-dentnode-com
```

### Agent turns time out (`TURN_TIMEOUT` error)

DeepSeek API may be slow or rate-limiting. Check DeepSeek status. To increase
the timeout (default 120 s), update the Cloud Run env var:
```bash
gcloud run services update laby-agent \
  --region=asia-south2 --project=app-dentnode-com \
  --set-env-vars=LABY_TURN_TIMEOUT=180
```

### Tool calls fail (`tool_error` in notes)

The agent could not reach the Node backend. Check:
1. `NODE_INTERNAL_BASE_URL` is set correctly (should be `https://app.dentnode.com/api`)
2. `INTERNAL_API_KEY` on this service matches the Node backend's value
3. Node backend `/internal/laby-tools/:tool` endpoints are healthy

### Canary health check fails in CI

The new revision failed `/health`. Check the Cloud Run logs for the candidate
revision. Common causes: missing secret version, bad `LABY_MODEL` value, or a
Python import error in the new code.

## Resource configuration (Cloud Run)

Current defaults used by the CI deploy command. Adjust if you see OOM crashes:

| Setting | Value |
|---------|-------|
| Memory | 512Mi (set via `--memory 512Mi` in deploy step if needed) |
| CPU | 1 |
| Min instances | 0 (scales to zero) |
| Max instances | 10 |
| Concurrency | 80 (Cloud Run default) |

To update:
```bash
gcloud run services update laby-agent \
  --region=asia-south2 --project=app-dentnode-com \
  --memory=1Gi --cpu=2
```

## Running tests locally

```bash
cd codebases/ai.dentnode.com
cp .env.example .env        # fill in real keys
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```

## First-time GCP setup checklist

- [ ] Create GCP project `app-dentnode-com` (or reuse existing)
- [ ] Enable Cloud Run API, Artifact Registry API, Secret Manager API
- [ ] Create Artifact Registry repo `dn-dashboard` in `asia-south2`
- [ ] Create secrets `LABY_INTERNAL_API_KEY` and `LABY_DEEPSEEK_API_KEY`
- [ ] Create a service account with roles: Cloud Run Admin, Artifact Registry Writer, Secret Manager Secret Accessor
- [ ] Add service account JSON as GitHub secret `GCP_SERVICE_ACCOUNT`
- [ ] Push to `main` to trigger first deploy
