from agno.run import RunContext

from . import storage
from .utils import clamp, now, truncate


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


def search_memories(run_context: RunContext, query: str, limit: int = 10) -> str:
    """Search semantic memories by keyword."""
    agent_id = run_context.session_state['agent_id']
    rows = storage.search_memories(agent_id=agent_id, query=query, limit=clamp(limit, 1, 50))
    if not rows:
        return 'No matching memories found.'

    storage.touch_memories(agent_id=agent_id, memory_ids=[row.id for row in rows], accessed_at=now())
    lines = []
    for row in rows:
        lines.append(
            f'- id={row.id} | name={row.name} | strength={row.strength}\n'
            f'  summary={row.summary}\n'
            f'  content={row.content}'
        )
    return 'Memories:\n' + '\n'.join(lines)


def create_memory(
    run_context: RunContext,
    name: str,
    summary: str,
    content: str,
    strength: int = 3,
) -> str:
    """Create a new semantic memory or replace one with the same name."""
    if strength < 0 or strength > 5:
        return 'Error: strength must be between 0 and 5.'

    agent_id = run_context.session_state['agent_id']
    storage.upsert_memory(
        agent_id=agent_id,
        name=name.strip(),
        summary=summary.strip(),
        content=content.strip(),
        strength=strength,
        updated_at=now(),
    )
    return f"Memory '{name}' saved."


def update_memory(
    run_context: RunContext,
    identifier: str,
    name: str | None = None,
    summary: str | None = None,
    content: str | None = None,
    strength: int | None = None,
) -> str:
    """Update an existing semantic memory by ID or name."""
    agent_id = run_context.session_state['agent_id']
    row = storage.find_memory(agent_id=agent_id, identifier=identifier)
    if row is None:
        return f"Memory '{identifier}' not found."

    next_name = name.strip() if name is not None else row.name
    next_summary = summary.strip() if summary is not None else row.summary
    next_content = content.strip() if content is not None else row.content
    next_strength = strength if strength is not None else row.strength
    if next_strength < 0 or next_strength > 5:
        return 'Error: strength must be between 0 and 5.'

    storage.update_memory(
        agent_id=agent_id,
        memory_id=row.id,
        name=next_name,
        summary=next_summary,
        content=next_content,
        strength=next_strength,
        updated_at=now(),
        metadata=row.metadata,
    )
    return f"Memory '{row.name}' updated."


def delete_memory(run_context: RunContext, identifier: str) -> str:
    """Delete a semantic memory by ID or name."""
    agent_id = run_context.session_state['agent_id']
    row = storage.find_memory(agent_id=agent_id, identifier=identifier)
    if row is None:
        return f"Memory '{identifier}' not found."

    storage.delete_memory(agent_id=agent_id, memory_id=row.id)
    return f"Memory '{row.name}' deleted."


def search_messages(run_context: RunContext, query: str, limit: int = 10) -> str:
    """Search through historical messages from past conversations."""
    agent_id = run_context.session_state['agent_id']
    rows = storage.search_messages(agent_id=agent_id, query=query, limit=clamp(limit, 1, 50))
    return _format_message_search_results(rows)


def run_shell(run_context: RunContext, command: str, timeout_seconds: int = 120) -> str:
    """Run a shell command inside this agent's isolated runtime workspace."""
    if timeout_seconds < 1 or timeout_seconds > 1800:
        return 'Error: timeout_seconds must be between 1 and 1800.'
    result = run_context.session_state['runtime'].exec(
        agent_id=run_context.session_state['agent_id'],
        command=command,
        timeout_seconds=timeout_seconds,
    )
    return truncate(result)


def list_workspace_files(run_context: RunContext, path: str = '.', limit: int = 200) -> str:
    """List files in the agent workspace."""
    if limit < 1 or limit > 1000:
        return 'Error: limit must be between 1 and 1000.'
    items = run_context.session_state['runtime'].list_files(agent_id=run_context.session_state['agent_id'], path=path)
    if not items:
        return 'No files found.'
    return 'Files:\n' + '\n'.join(f'- {item}' for item in items[:limit])


def read_workspace_file(run_context: RunContext, path: str, max_chars: int = 20_000) -> str:
    """Read a UTF-8 text file from the agent workspace."""
    if max_chars < 1 or max_chars > 200_000:
        return 'Error: max_chars must be between 1 and 200000.'
    try:
        content = run_context.session_state['runtime'].read_file(agent_id=run_context.session_state['agent_id'], path=path)
    except Exception as exc:
        return f'Error: {exc}'
    return truncate(content, limit=max_chars)


def write_workspace_file(run_context: RunContext, path: str, content: str) -> str:
    """Write a UTF-8 text file to the agent workspace."""
    try:
        written = run_context.session_state['runtime'].write_file(agent_id=run_context.session_state['agent_id'], path=path, content=content)
    except Exception as exc:
        return f'Error: {exc}'
    return f'Wrote file: {written}'


def delete_workspace_path(run_context: RunContext, path: str) -> str:
    """Delete a file or folder from the agent workspace."""
    try:
        deleted = run_context.session_state['runtime'].delete_path(agent_id=run_context.session_state['agent_id'], path=path)
    except Exception as exc:
        return f'Error: {exc}'
    if not deleted:
        return f'Path not found: {path}'
    return f'Deleted: {path}'


TOOLS = [
    search_memories,
    create_memory,
    update_memory,
    delete_memory,
    search_messages,
    run_shell,
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
    delete_workspace_path,
]
