from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import storage
from .config import CONFIG


@dataclass(frozen=True)
class RuntimeInfo:
    mode: str  # docker | local
    workspace: Path
    container_name: str | None = None


class RuntimeManager:
    def __init__(
        self,
        *,
        base_dir: str | None = None,
        docker_enabled: bool | None = None,
        docker_image: str | None = None,
    ):
        resolved_base = base_dir or CONFIG.runtime.base_dir
        self._base_dir = Path(resolved_base)
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback = Path('/tmp/mico_agents')
            fallback.mkdir(parents=True, exist_ok=True)
            self._base_dir = fallback
        self._docker_enabled = CONFIG.runtime.docker_enabled if docker_enabled is None else docker_enabled
        self._docker_image = docker_image or CONFIG.runtime.docker_image
        self._lock = threading.RLock()

    def workspace_path(self, agent_id: str) -> Path:
        return self._base_dir / agent_id / 'workspace'

    def container_name(self, agent_id: str) -> str:
        safe = ''.join(ch for ch in agent_id if ch.isalnum() or ch in {'-', '_'})
        return f'mico-agent-{safe[:32]}'

    def ensure_running(self, agent_id: str) -> RuntimeInfo:
        with self._lock:
            workspace = self.workspace_path(agent_id)
            workspace.mkdir(parents=True, exist_ok=True)

            if not self._docker_enabled:
                info = RuntimeInfo(mode='local', workspace=workspace, container_name=None)
                self._persist_runtime_info(agent_id, info)
                return info

            name = self.container_name(agent_id)
            inspect = self._run(['docker', 'inspect', '-f', '{{.State.Running}}', name], check=False)
            if inspect.returncode != 0:
                self._run(
                    [
                        'docker',
                        'run',
                        '--detach',
                        '--name',
                        name,
                        '--workdir',
                        '/workspace',
                        '--volume',
                        f'{workspace.resolve()}:/workspace',
                        '--network',
                        'none',
                        '--memory',
                        '512m',
                        '--cpus',
                        '1.0',
                        self._docker_image,
                        'sleep',
                        'infinity',
                    ]
                )
            elif inspect.stdout.strip().lower() != 'true':
                self._run(['docker', 'start', name])

            info = RuntimeInfo(mode='docker', workspace=workspace, container_name=name)
            self._persist_runtime_info(agent_id, info)
            return info

    def stop(self, agent_id: str) -> bool:
        with self._lock:
            if not self._docker_enabled:
                self._update_runtime_fields(agent_id, running=False, last_stopped_at=int(time.time()))
                return False

            name = self.container_name(agent_id)
            inspect = self._run(['docker', 'inspect', '-f', '{{.State.Running}}', name], check=False)
            if inspect.returncode != 0:
                self._update_runtime_fields(agent_id, running=False, last_stopped_at=int(time.time()))
                return False

            running = inspect.stdout.strip().lower() == 'true'
            if running:
                self._run(['docker', 'stop', name], check=False)
            self._update_runtime_fields(agent_id, running=False, last_stopped_at=int(time.time()))
            return running

    def status(self, agent_id: str) -> dict[str, object]:
        workspace = self.workspace_path(agent_id)
        workspace.mkdir(parents=True, exist_ok=True)
        row = storage.get_agent(agent_id)
        runtime_payload = dict(row.runtime) if row is not None else {}
        mode = str(runtime_payload.get('mode') or ('docker' if self._docker_enabled else 'local'))

        state = 'unknown'
        running = False
        container_name = self.container_name(agent_id) if mode == 'docker' else None

        if mode == 'docker' and self._docker_enabled:
            inspect = self._run(['docker', 'inspect', '-f', '{{.State.Status}}', container_name or ''], check=False)
            if inspect.returncode != 0:
                state = 'missing'
                running = False
            else:
                state = inspect.stdout.strip().lower() or 'unknown'
                running = state == 'running'
        else:
            state = 'ready'
            running = False

        return {
            'mode': mode,
            'state': state,
            'running': running,
            'workspace': str(workspace.resolve()),
            'container_name': container_name,
            'image': self._docker_image if mode == 'docker' else None,
            'meta': runtime_payload,
        }

    def exec(self, *, agent_id: str, command: str, timeout_seconds: int = 120) -> str:
        info = self.ensure_running(agent_id)
        if info.mode == 'local':
            completed = self._run(
                ['sh', '-lc', command],
                cwd=info.workspace,
                timeout=timeout_seconds,
                check=False,
            )
        else:
            completed = self._run(
                ['docker', 'exec', info.container_name or '', 'sh', '-lc', command],
                timeout=timeout_seconds,
                check=False,
            )

        output = '\n'.join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
        if completed.returncode != 0:
            return (
                f'Command failed with exit code {completed.returncode}.\n'
                f'{output or "No output."}'
            )
        return output or '(no output)'

    def list_files(self, *, agent_id: str, path: str = '.') -> list[str]:
        workspace = self.workspace_path(agent_id)
        workspace.mkdir(parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)
        workspace_resolved = workspace.resolve()
        if not target.exists():
            return []
        if target.is_file():
            return [str(target.relative_to(workspace_resolved))]

        items: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            rel = child.relative_to(workspace_resolved)
            suffix = '/' if child.is_dir() else ''
            items.append(f'{rel}{suffix}')
        return items

    def read_file(self, *, agent_id: str, path: str) -> str:
        workspace = self.workspace_path(agent_id)
        workspace.mkdir(parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)
        if not target.exists() or not target.is_file():
            raise ValueError(f"File '{path}' not found in agent workspace.")
        return target.read_text(encoding='utf-8')

    def write_file(self, *, agent_id: str, path: str, content: str) -> str:
        workspace = self.workspace_path(agent_id)
        workspace.mkdir(parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return str(target.relative_to(workspace.resolve()))

    def delete_path(self, *, agent_id: str, path: str) -> bool:
        workspace = self.workspace_path(agent_id)
        workspace.mkdir(parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)
        if not target.exists():
            return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return True

    def delete_agent_runtime(self, agent_id: str) -> None:
        with self._lock:
            if self._docker_enabled:
                name = self.container_name(agent_id)
                self._run(['docker', 'rm', '-f', name], check=False)

            workspace_root = self.workspace_path(agent_id).parent
            if workspace_root.exists():
                shutil.rmtree(workspace_root)

    def _persist_runtime_info(self, agent_id: str, info: RuntimeInfo) -> None:
        row = storage.get_agent(agent_id)
        if row is None:
            return

        runtime_payload = dict(row.runtime)
        runtime_payload.update(
            {
                'mode': info.mode,
                'workspace': str(info.workspace.resolve()),
                'container_name': info.container_name,
                'image': self._docker_image if info.mode == 'docker' else None,
                'running': info.mode == 'docker',
                'last_seen': int(time.time()),
            }
        )
        storage.update_agent(
            agent_id=agent_id,
            runtime=runtime_payload,
            updated_at=int(time.time()),
        )

    def update_runtime_meta(self, *, agent_id: str, **fields: object) -> None:
        self._update_runtime_fields(agent_id, **fields)

    def _update_runtime_fields(self, agent_id: str, **fields: object) -> None:
        row = storage.get_agent(agent_id)
        if row is None:
            return
        runtime_payload = dict(row.runtime)
        runtime_payload.update(fields)
        runtime_payload['last_seen'] = int(time.time())
        storage.update_agent(agent_id=agent_id, runtime=runtime_payload, updated_at=int(time.time()))

    @staticmethod
    def _run(
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 120,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            stderr = (completed.stderr or '').strip()
            stdout = (completed.stdout or '').strip()
            detail = stderr or stdout or 'no output'
            raise RuntimeError(f"Command failed ({' '.join(args)}): {detail}")
        return completed

    @staticmethod
    def _resolve_workspace_path(workspace: Path, raw_path: str) -> Path:
        target = (workspace / raw_path).resolve()
        workspace_resolved = workspace.resolve()
        if workspace_resolved == target or workspace_resolved in target.parents:
            return target
        raise ValueError(f"Path '{raw_path}' escapes the agent workspace.")


_RUNTIME_MANAGER: RuntimeManager | None = None


def get_runtime_manager() -> RuntimeManager:
    global _RUNTIME_MANAGER
    if _RUNTIME_MANAGER is None:
        _RUNTIME_MANAGER = RuntimeManager()
    return _RUNTIME_MANAGER


def reset_runtime_manager() -> None:
    global _RUNTIME_MANAGER
    _RUNTIME_MANAGER = None
