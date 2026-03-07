import asyncio
import json
import re
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .runtime import get_runtime_manager


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    agent_id: str
    name: str
    summary: str
    content: str
    strength: int
    created_at: str
    updated_at: str


def _dir(agent_id: str) -> Path:
    return get_runtime_manager().workspace_path(agent_id) / 'memories'


def _slug(value: str) -> str:
    value = re.sub(r'[^a-z0-9]+', '-', value.strip().lower()).strip('-')
    return value or 'memory'


def _path(agent_id: str, name: str, memory_id: str) -> Path:
    return _dir(agent_id) / f'{_slug(name)}--{memory_id}.md'


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _utc(value: object) -> str:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Memory timestamps must include timezone information.')
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _parse(text: str) -> tuple[dict, str]:
    if not text.startswith('+++\n'):
        raise ValueError('Missing TOML front matter.')
    try:
        frontmatter, body = text[4:].split('\n+++\n', 1)
    except ValueError as exc:
        raise ValueError('Unclosed TOML front matter.') from exc
    return tomllib.loads(frontmatter), body.strip()


def _read(agent_id: str, path: Path) -> MemoryRecord:
    data, content = _parse(path.read_text(encoding='utf-8'))
    return MemoryRecord(
        id=str(data['id']).strip(),
        agent_id=agent_id,
        name=str(data['name']).strip(),
        summary=str(data['summary']).strip(),
        content=content,
        strength=int(data['strength']),
        created_at=_utc(data['created_at']),
        updated_at=_utc(data['updated_at']),
    )


def _list(agent_id: str) -> list[tuple[Path, MemoryRecord]]:
    rows: list[tuple[Path, MemoryRecord]] = []
    for path in sorted(_dir(agent_id).glob('*.md')):
        try:
            rows.append((path, _read(agent_id, path)))
        except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
            continue
    return rows


def _write(path: Path, row: MemoryRecord) -> None:
    lines = [
        '+++',
        f'id = {json.dumps(row.id, ensure_ascii=True)}',
        f'name = {json.dumps(row.name, ensure_ascii=True)}',
        f'summary = {json.dumps(row.summary, ensure_ascii=True)}',
        f'strength = {row.strength}',
        f'created_at = {row.created_at}',
        f'updated_at = {row.updated_at}',
        '+++',
        '',
        row.content.strip(),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    tmp.replace(path)


async def find_memory(*, agent_id: str, identifier: str) -> MemoryRecord | None:
    rows = await asyncio.to_thread(_list, agent_id)
    return next((row for _, row in rows if identifier in {row.id, row.name}), None)


async def upsert_memory(
    *,
    agent_id: str,
    name: str,
    summary: str,
    content: str,
    strength: int,
) -> None:
    rows = await asyncio.to_thread(_list, agent_id)
    existing = next(((path, row) for path, row in rows if row.name == name), None)
    now = _now()
    row = MemoryRecord(
        id=existing[1].id if existing else str(uuid.uuid4()),
        agent_id=agent_id,
        name=name,
        summary=summary,
        content=content,
        strength=strength,
        created_at=existing[1].created_at if existing else now,
        updated_at=now,
    )
    path = _path(agent_id, row.name, row.id)
    await asyncio.to_thread(_write, path, row)
    if existing and existing[0] != path and existing[0].exists():
        await asyncio.to_thread(existing[0].unlink)


async def update_memory(
    *,
    agent_id: str,
    memory_id: str,
    name: str,
    summary: str,
    content: str,
    strength: int,
) -> None:
    rows = await asyncio.to_thread(_list, agent_id)
    current = next(((path, row) for path, row in rows if row.id == memory_id), None)
    if current is None:
        return
    if any(row.id != memory_id and row.name == name for _, row in rows):
        raise ValueError(f"Memory '{name}' already exists.")
    row = MemoryRecord(
        id=current[1].id,
        agent_id=agent_id,
        name=name,
        summary=summary,
        content=content,
        strength=strength,
        created_at=current[1].created_at,
        updated_at=_now(),
    )
    path = _path(agent_id, row.name, row.id)
    await asyncio.to_thread(_write, path, row)
    if current[0] != path and current[0].exists():
        await asyncio.to_thread(current[0].unlink)


async def delete_memory(*, agent_id: str, memory_id: str) -> None:
    rows = await asyncio.to_thread(_list, agent_id)
    path = next((path for path, row in rows if row.id == memory_id), None)
    if path is not None and path.exists():
        await asyncio.to_thread(path.unlink)


async def search_memories(*, agent_id: str, query: str, limit: int) -> list[MemoryRecord]:
    needle = query.strip().lower()
    rows = [row for _, row in await asyncio.to_thread(_list, agent_id)]
    if needle:
        rows = [row for row in rows if needle in '\n'.join((row.name, row.summary, row.content)).lower()]
    return sorted(rows, key=lambda row: (row.strength, row.updated_at), reverse=True)[:limit]
