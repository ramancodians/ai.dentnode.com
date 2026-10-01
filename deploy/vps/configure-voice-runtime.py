"""Provision the existing VPS deployment for the optional voice worker.

Run as root after the supervisor image has deployed with voice disabled. This
edits configuration only; deploy/redeploy application images via GitHub Actions.
Provider credentials remain on the server and are never printed.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4
from datetime import datetime, timezone

BASE = Path('/opt/dentnode/apps/ai.dentnode.com')
SOURCE = Path('/opt/dentnode/apps/calling.dentnode.com/env/secrets.env')


def read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != 0:
        raise RuntimeError(f'Expected root-owned regular file: {path}')
    return path.read_text()


def env_update(text, values):
    lines = text.splitlines()
    for key, value in values.items():
        if '\n' in value or '\r' in value:
            raise ValueError('Multiline configuration is not supported')
        positions = [i for i, line in enumerate(lines) if line.startswith(key + '=')]
        if len(positions) > 1:
            raise ValueError(f'Duplicate configuration key: {key}')
        if positions:
            lines[positions[0]] = key + '=' + value
        else:
            lines.append(key + '=' + value)
    return '\n'.join(lines) + '\n'


def atomic_write(path, content, mode):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.voice-')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    if os.geteuid() != 0:
        raise RuntimeError('Root is required to provision runtime configuration')
    state = dict(line.split('=', 1) for line in read(BASE / 'state/release-state').splitlines() if '=' in line)
    slot = state.get('active_slot')
    if slot not in ('blue', 'green'):
        raise RuntimeError('No active AI deployment')
    # The next rollout and rollback must both have the supervisor available.
    subprocess.run(['docker', 'exec', f'dentnode-ai-{slot}', 'python', '-c',
                    'import runtime; assert callable(runtime.health)'], check=True)
    source = dict(line.split('=', 1) for line in read(SOURCE).splitlines() if '=' in line and not line.startswith('#'))
    keys = ('LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET')
    if any(not source.get(key) for key in keys):
        raise RuntimeError('Calling Service is missing LiveKit configuration')
    secrets = BASE / 'env/secrets.env'
    runtime = BASE / 'env/runtime.env'
    release = BASE / 'env/release.env'
    compose = BASE / 'compose.yaml'
    original = {p: read(p) for p in (secrets, runtime, release, compose)}
    compose_text, replaced = re.subn(r'(?m)^    test: \["CMD", "python", "-c", .*\]$',
                                   '    test: ["CMD", "python", "-m", "runtime", "--health"]', original[compose])
    if replaced != 1 and 'test: ["CMD", "python", "-m", "runtime", "--health"]' not in compose_text:
        raise RuntimeError('Unexpected compose healthcheck; review before applying')
    compose_text, grace_replaced = re.subn(r'(?m)^  stop_grace_period: \d+s$', '  stop_grace_period: 360s', compose_text)
    if grace_replaced != 1:
        raise RuntimeError('Unexpected compose stop grace; review before applying')
    proposed = {
        secrets: env_update(original[secrets], {key: source[key] for key in keys}),
        runtime: env_update(original[runtime], {'VOICE_WORKER_ENABLED': 'true',
            'LIVEKIT_AGENT_NAME': 'dentnode-receptionist', 'VOICE_WORKER_IDLE_PROCESSES': '1',
            'CALLING_SERVICE_URL': 'https://calling.dentnode.com'}),
        release: env_update(original[release], {'AI_MEMORY_LIMIT': '2g'}),
        compose: compose_text,
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:12]
    backups = {}
    try:
        for path, content in proposed.items():
            backup = path.with_name(path.name + '.pre-inbound-ai-' + stamp)
            shutil.copy2(path, backup)
            os.chmod(backup, 0o600)
            backups[path] = backup
            atomic_write(path, content, 0o600 if path == secrets else path.stat().st_mode & 0o777)
        validation = subprocess.run(['docker', 'compose', '--env-file', str(release), '-f', str(compose),
                        'config', '--quiet'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if validation.returncode:
            raise RuntimeError('Compose configuration validation failed; original configuration restored')
    except Exception:
        for path, backup in backups.items():
            atomic_write(path, original[path], 0o600 if path == secrets else path.stat().st_mode & 0o777)
        raise
    print(json.dumps({'configured': True, 'worker_started': False,
                      'next_step': 'Redeploy the reviewed supervisor image using GitHub Actions',
                      'backup_suffix': '.pre-inbound-ai-' + stamp}))


if __name__ == '__main__':
    main()
