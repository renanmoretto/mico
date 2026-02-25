from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import storage
from .config import CONFIG

logger = logging.getLogger(__name__)


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
        self._lock = asyncio.Lock()

    def workspace_path(self, agent_id: str) -> Path:
        return self._base_dir / agent_id / 'workspace'

    def container_name(self, agent_id: str) -> str:
        safe = ''.join(ch for ch in agent_id if ch.isalnum() or ch in {'-', '_'})
        return f'mico-agent-{safe[:32]}'

    async def ensure_running(self, agent_id: str) -> RuntimeInfo:
        async with self._lock:
            workspace = self.workspace_path(agent_id)
            await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)

            if not self._docker_enabled:
                info = RuntimeInfo(mode='local', workspace=workspace, container_name=None)
                await self._persist_runtime_info(agent_id, info)
                return info

            name = self.container_name(agent_id)
            inspect = await self._run(['docker', 'inspect', '-f', '{{.State.Running}}', name], check=False)
            if inspect.returncode != 0:
                await self._run(
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
                await self._run(['docker', 'start', name])

            info = RuntimeInfo(mode='docker', workspace=workspace, container_name=name)
            await self._persist_runtime_info(agent_id, info)
            return info

    async def stop(self, agent_id: str) -> bool:
        async with self._lock:
            if not self._docker_enabled:
                await self._update_runtime_fields(agent_id, running=False, last_stopped_at=int(time.time()))
                return False

            name = self.container_name(agent_id)
            inspect = await self._run(['docker', 'inspect', '-f', '{{.State.Running}}', name], check=False)
            if inspect.returncode != 0:
                await self._update_runtime_fields(agent_id, running=False, last_stopped_at=int(time.time()))
                return False

            running = inspect.stdout.strip().lower() == 'true'
            if running:
                await self._run(['docker', 'stop', name], check=False)
            await self._update_runtime_fields(agent_id, running=False, last_stopped_at=int(time.time()))
            return running

    async def status(self, agent_id: str) -> dict[str, object]:
        workspace = self.workspace_path(agent_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        row = await storage.get_agent(agent_id)
        runtime_payload = dict(row.runtime) if row is not None else {}
        mode = str(runtime_payload.get('mode') or ('docker' if self._docker_enabled else 'local'))

        state = 'unknown'
        running = False
        container_name = self.container_name(agent_id) if mode == 'docker' else None

        if mode == 'docker' and self._docker_enabled:
            inspect = await self._run(['docker', 'inspect', '-f', '{{.State.Status}}', container_name or ''], check=False)
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

    async def exec(self, *, agent_id: str, command: str, timeout_seconds: int = 120) -> str:
        info = await self.ensure_running(agent_id)
        if info.mode == 'local':
            completed = await self._run(
                ['sh', '-lc', command],
                cwd=info.workspace,
                timeout=timeout_seconds,
                check=False,
            )
        else:
            completed = await self._run(
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

    async def list_files(self, *, agent_id: str, path: str = '.') -> list[str]:
        workspace = self.workspace_path(agent_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)
        workspace_resolved = workspace.resolve()

        def _list() -> list[str]:
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

        return await asyncio.to_thread(_list)

    async def read_file(self, *, agent_id: str, path: str) -> str:
        workspace = self.workspace_path(agent_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)

        def _read() -> str:
            if not target.exists() or not target.is_file():
                raise ValueError(f"File '{path}' not found in agent workspace.")
            return target.read_text(encoding='utf-8')

        return await asyncio.to_thread(_read)

    async def write_file(self, *, agent_id: str, path: str, content: str) -> str:
        workspace = self.workspace_path(agent_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)

        def _write() -> str:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return str(target.relative_to(workspace.resolve()))

        return await asyncio.to_thread(_write)

    async def delete_path(self, *, agent_id: str, path: str) -> bool:
        workspace = self.workspace_path(agent_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        target = self._resolve_workspace_path(workspace, path)

        def _delete() -> bool:
            if not target.exists():
                return False
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return True

        return await asyncio.to_thread(_delete)

    async def cleanup_orphaned_containers(self) -> list[str]:
        """Remove Docker containers that don't match any agent in the DB."""
        if not self._docker_enabled:
            return []

        result = await self._run(
            ['docker', 'ps', '-a', '--filter', 'name=mico-agent-', '--format', '{{.Names}}'],
            check=False,
        )
        if result.returncode != 0:
            logger.warning('Failed to list Docker containers for cleanup: %s', result.stderr.strip())
            return []
        if not result.stdout.strip():
            return []

        # Docker name filter is substring-based, so verify prefix
        container_names = [
            n for n in result.stdout.strip().splitlines()
            if n.startswith('mico-agent-')
        ]

        agents = await storage.list_agents()
        known_names = {self.container_name(a.id) for a in agents}

        removed: list[str] = []
        for name in container_names:
            if name not in known_names:
                await self._run(['docker', 'rm', '-f', name], check=False)
                removed.append(name)
        return removed

    async def delete_agent_runtime(self, agent_id: str) -> None:
        async with self._lock:
            if self._docker_enabled:
                name = self.container_name(agent_id)
                await self._run(['docker', 'rm', '-f', name], check=False)

            workspace_root = self.workspace_path(agent_id).parent

            def _remove() -> None:
                if workspace_root.exists():
                    shutil.rmtree(workspace_root)

            await asyncio.to_thread(_remove)

    async def _persist_runtime_info(self, agent_id: str, info: RuntimeInfo) -> None:
        row = await storage.get_agent(agent_id)
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
        await storage.update_agent(
            agent_id=agent_id,
            runtime=runtime_payload,
            updated_at=int(time.time()),
        )

    async def update_runtime_meta(self, *, agent_id: str, **fields: object) -> None:
        await self._update_runtime_fields(agent_id, **fields)

    async def _update_runtime_fields(self, agent_id: str, **fields: object) -> None:
        row = await storage.get_agent(agent_id)
        if row is None:
            return
        runtime_payload = dict(row.runtime)
        runtime_payload.update(fields)
        runtime_payload['last_seen'] = int(time.time())
        await storage.update_agent(agent_id=agent_id, runtime=runtime_payload, updated_at=int(time.time()))

    @dataclass(frozen=True)
    class _ProcessResult:
        returncode: int
        stdout: str
        stderr: str

    @staticmethod
    async def _run(
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 120,
        check: bool = True,
    ) -> RuntimeManager._ProcessResult:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise

        stdout = (stdout_bytes or b'').decode('utf-8', errors='replace')
        stderr = (stderr_bytes or b'').decode('utf-8', errors='replace')
        returncode = process.returncode or 0

        if check and returncode != 0:
            detail = stderr.strip() or stdout.strip() or 'no output'
            raise RuntimeError(f"Command failed ({' '.join(args)}): {detail}")

        return RuntimeManager._ProcessResult(returncode=returncode, stdout=stdout, stderr=stderr)

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
