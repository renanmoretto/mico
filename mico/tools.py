import time

from . import crons, storage


def _now() -> int:
    return int(time.time())


def _clamp(limit: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(limit, maximum))


def _format_message_search_results(rows: list[storage.MessageRecord]) -> str:
    if not rows:
        return 'No matching messages found.'

    lines = []
    for row in rows:
        lines.append(
            f'- id={row.id} | timestamp={row.timestamp} | role={row.role}\n'
            f'  content={row.content}'
        )
    return 'Messages:\n' + '\n'.join(lines)


def search_memories(query: str, limit: int = 10) -> str:
    """Search semantic memories by keyword."""
    rows = storage.search_memories(query=query, limit=_clamp(limit, 1, 50))
    if not rows:
        return 'No matching memories found.'

    storage.touch_memories(memory_ids=[row.id for row in rows], accessed_at=_now())
    lines = []
    for row in rows:
        lines.append(
            f'- id={row.id} | name={row.name} | strength={row.strength}\n'
            f'  summary={row.summary}\n'
            f'  content={row.content}'
        )
    return 'Memories:\n' + '\n'.join(lines)


def create_memory(
    name: str,
    summary: str,
    content: str,
    strength: int = 3,
) -> str:
    """Create a new semantic memory or replace one with the same name."""
    if strength < 0 or strength > 5:
        return 'Error: strength must be between 0 and 5.'

    storage.upsert_memory(
        name=name.strip(),
        summary=summary.strip(),
        content=content.strip(),
        strength=strength,
        updated_at=_now(),
    )
    return f"Memory '{name}' saved."


def update_memory(
    identifier: str,
    name: str | None = None,
    summary: str | None = None,
    content: str | None = None,
    strength: int | None = None,
) -> str:
    """Update an existing semantic memory by ID or name."""
    row = storage.find_memory(identifier)
    if row is None:
        return f"Memory '{identifier}' not found."

    next_name = name.strip() if name is not None else row.name
    next_summary = summary.strip() if summary is not None else row.summary
    next_content = content.strip() if content is not None else row.content
    next_strength = strength if strength is not None else row.strength
    if next_strength < 0 or next_strength > 5:
        return 'Error: strength must be between 0 and 5.'

    storage.update_memory(
        memory_id=row.id,
        name=next_name,
        summary=next_summary,
        content=next_content,
        strength=next_strength,
        updated_at=_now(),
    )
    return f"Memory '{row.name}' updated."


def delete_memory(identifier: str) -> str:
    """Delete a semantic memory by ID or name."""
    row = storage.find_memory(identifier)
    if row is None:
        return f"Memory '{identifier}' not found."

    storage.delete_memory(memory_id=row.id)
    return f"Memory '{row.name}' deleted."


def search_messages(query: str, limit: int = 10) -> str:
    """Search through historical messages from past conversations."""
    rows = storage.search_messages(query=query, limit=_clamp(limit, 1, 50))
    return _format_message_search_results(rows)


def create_cron(name: str, prompt: str, when: str) -> str:
    """Create a one-time cron job for a future reminder/task."""
    return crons.create_cron(name=name, prompt=prompt, when=when)


def list_crons(limit: int = 20, include_done: bool = False) -> str:
    """List cron jobs."""
    return crons.list_crons(limit=limit, include_done=include_done)


def delete_cron(identifier: str) -> str:
    """Delete a cron job by ID."""
    return crons.delete_cron(identifier=identifier)


TOOLS = [
    search_memories,
    create_memory,
    update_memory,
    delete_memory,
    search_messages,
    create_cron,
    list_crons,
    delete_cron,
]
