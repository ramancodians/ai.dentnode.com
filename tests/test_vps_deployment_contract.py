"""Static guardrails for the VPS image and delivery contract."""

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_vps_workflow_is_digest_only_and_actions_are_commit_pinned():
    workflow = (ROOT / ".github/workflows/deploy-vps.yaml").read_text()

    uses = re.findall(r"^\s*uses:\s*([^\s#]+)", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", item) for item in uses)
    assert re.search(r"^  push:\n    branches: \[main\]$", workflow, flags=re.MULTILINE)
    assert "workflow_dispatch:" in workflow
    assert "workflow_run:" not in workflow
    assert "github.event.workflow_run" not in workflow
    assert "deploy-auth $SERVICE_ID $image_ref $GHCR_ACTOR" in workflow
    assert "printf '%s' \"$GHCR_TOKEN\" | ssh" in workflow
    assert "ssh-keyscan" not in workflow
    assert "IMAGE_DIGEST" in workflow
    assert "sha256:[0-9a-f]{64}" in workflow


def test_cloud_run_deployment_is_removed():
    assert not (ROOT / ".github/workflows/cloud-run-deploy.yaml").exists()


def test_pull_request_ci_is_read_only_and_has_no_deployment_credentials():
    workflow = (ROOT / ".github/workflows/ci.yaml").read_text()

    uses = re.findall(r"^\s*uses:\s*([^\s#]+)", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", item) for item in uses)
    assert "pull_request:" in workflow
    assert "contents: read" in workflow
    assert "packages: write" not in workflow
    assert "secrets." not in workflow
    assert "docker" not in workflow.lower()
    assert "ssh" not in workflow.lower()


def test_vps_compose_has_no_public_port_or_data_network():
    compose = yaml.safe_load((ROOT / "deploy/vps/compose.yaml").read_text())
    service = compose["services"]["ai"]

    assert service["image"].startswith("${AI_IMAGE:?")
    assert service["user"] == "10001:10001"
    assert service["read_only"] is True
    assert "ports" not in service
    assert service["expose"] == ["8080"]
    assert set(service["networks"]) == {"dentnode-edge", "dentnode-telemetry"}
    assert "dentnode-data" not in compose["networks"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["cap_drop"] == ["ALL"]
    assert service["healthcheck"]["test"][0] == "CMD"


def test_production_image_runs_nonroot_and_has_an_internal_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "USER laby" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:" in dockerfile
    assert "install -d -o laby -g laby -m 0700 /var/lib/dentnode-ai" in dockerfile


def test_caddy_caps_call_audio_body_before_reverse_proxying():
    caddy = (ROOT / "deploy/vps/caddy-site.template").read_text()
    runtime_env = (ROOT / "deploy/vps/env/runtime.env.example").read_text()

    assert "@callAudio path /internal/call-audio/analyze" in caddy
    assert re.search(
        r"request_body\s+@callAudio\s*\{\s*max_size\s+27MB\s*\}",
        caddy,
        flags=re.MULTILINE,
    )
    assert "CALL_AUDIO_REQUEST_MAX_BYTES=27000000" in runtime_env
