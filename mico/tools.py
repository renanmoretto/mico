from datetime import datetime, timezone

from agno.run import RunContext
from croniter import croniter

from . import storage
from .messages import OutboundMessage
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


async def search_memories(run_context: RunContext, query: str, limit: int = 10) -> str:
    """Search your memories by keyword.

    Matches against name, summary, and content fields.
    Always search before creating new memories to avoid duplicates.
    Results are ranked by strength (highest first), then by most recently updated.
    Limit: 1-50 (default 10).
    """
    agent_id = run_context.session_state['agent_id']
    rows = await storage.search_memories(agent_id=agent_id, query=query, limit=clamp(limit, 1, 50))
    if not rows:
        return 'No matching memories found.'

    await storage.touch_memories(agent_id=agent_id, memory_ids=[row.id for row in rows], accessed_at=now())
    lines = []
    for row in rows:
        lines.append(
            f'- id={row.id} | name={row.name} | strength={row.strength}\n'
            f'  summary={row.summary}\n'
            f'  content={row.content}'
        )
    return 'Memories:\n' + '\n'.join(lines)


async def create_memory(
    run_context: RunContext,
    name: str,
    summary: str,
    content: str,
    strength: int = 3,
) -> str:
    """Create a new memory or replace an existing one with the same name.

    Use memories to persist important facts, user preferences, learned context, or
    anything worth remembering across conversations.

    Fields:
    - name: unique identifier for the memory. If a memory with this name already exists, it will be replaced.
    - summary: a short one-line description of the memory.
    - content: the full detailed content of the memory.
    - strength: importance level from 0 to 5 (default 3). Memories are ranked by strength on search,
      so higher-strength memories surface first. Use 5 for critical/permanent knowledge (e.g. core user
      preferences, key facts), 3 for normal information, and 0-1 for low-priority or ephemeral notes.
    """
    if strength < 0 or strength > 5:
        return 'Error: strength must be between 0 and 5.'

    agent_id = run_context.session_state['agent_id']
    await storage.upsert_memory(
        agent_id=agent_id,
        name=name.strip(),
        summary=summary.strip(),
        content=content.strip(),
        strength=strength,
        updated_at=now(),
    )
    return f"Memory '{name}' saved."


async def update_memory(
    run_context: RunContext,
    identifier: str,
    name: str | None = None,
    summary: str | None = None,
    content: str | None = None,
    strength: int | None = None,
) -> str:
    """Update an existing memory by its ID or name.

    Only provided fields are changed; omitted fields keep their current values.
    Use this to refine, correct, or re-prioritize stored knowledge.

    Strength (0-5) controls how prominently this memory appears in search results:
    5 = critical/always surface first, 3 = normal, 0-1 = low priority.
    Increase strength for memories that prove important; decrease for less relevant ones.
    """
    agent_id = run_context.session_state['agent_id']
    row = await storage.find_memory(agent_id=agent_id, identifier=identifier)
    if row is None:
        return f"Memory '{identifier}' not found."

    next_name = name.strip() if name is not None else row.name
    next_summary = summary.strip() if summary is not None else row.summary
    next_content = content.strip() if content is not None else row.content
    next_strength = strength if strength is not None else row.strength
    if next_strength < 0 or next_strength > 5:
        return 'Error: strength must be between 0 and 5.'

    await storage.update_memory(
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


async def delete_memory(run_context: RunContext, identifier: str) -> str:
    """Permanently delete a memory by its ID or name.

    Use this to remove outdated, incorrect, or no longer relevant information.
    Consider updating the memory instead if it just needs correction.
    """
    agent_id = run_context.session_state['agent_id']
    row = await storage.find_memory(agent_id=agent_id, identifier=identifier)
    if row is None:
        return f"Memory '{identifier}' not found."

    await storage.delete_memory(agent_id=agent_id, memory_id=row.id)
    return f"Memory '{row.name}' deleted."


async def search_messages(run_context: RunContext, query: str, limit: int = 10) -> str:
    """Search through past conversation messages by keyword.

    Returns matching messages with their timestamps and roles (user/assistant).
    Useful for recalling what was discussed in previous conversations.
    Results are ordered by most recent first.
    Limit: 1-50 (default 10).
    """
    agent_id = run_context.session_state['agent_id']
    rows = await storage.search_messages(agent_id=agent_id, query=query, limit=clamp(limit, 1, 50))
    return _format_message_search_results(rows)


async def run_shell(run_context: RunContext, command: str, timeout_seconds: int = 120) -> str:
    """Run a shell command in your isolated workspace container.

    Use this for installing packages, running scripts, processing data, or any
    command-line task. The workspace filesystem persists across calls.
    Timeout: 1-1800 seconds (default 120). Output is truncated if too long.
    """
    if timeout_seconds < 1 or timeout_seconds > 1800:
        return 'Error: timeout_seconds must be between 1 and 1800.'
    result = await run_context.session_state['runtime'].exec(
        agent_id=run_context.session_state['agent_id'],
        command=command,
        timeout_seconds=timeout_seconds,
    )
    return truncate(result)


async def list_workspace_files(run_context: RunContext, path: str = '.', limit: int = 200) -> str:
    """List files and directories in your workspace.

    Path is relative to the workspace root (use '.' for root).
    Limit: 1-1000 entries (default 200).
    """
    if limit < 1 or limit > 1000:
        return 'Error: limit must be between 1 and 1000.'
    items = await run_context.session_state['runtime'].list_files(agent_id=run_context.session_state['agent_id'], path=path)
    if not items:
        return 'No files found.'
    return 'Files:\n' + '\n'.join(f'- {item}' for item in items[:limit])


async def read_workspace_file(run_context: RunContext, path: str, max_chars: int = 20_000) -> str:
    """Read a text file from your workspace.

    Path is relative to workspace root.
    Output is truncated at max_chars (1-200000, default 20000).
    Use this to inspect file contents before modifying them.
    """
    if max_chars < 1 or max_chars > 200_000:
        return 'Error: max_chars must be between 1 and 200000.'
    try:
        content = await run_context.session_state['runtime'].read_file(agent_id=run_context.session_state['agent_id'], path=path)
    except Exception as exc:
        return f'Error: {exc}'
    return truncate(content, limit=max_chars)


async def write_workspace_file(run_context: RunContext, path: str, content: str) -> str:
    """Write or overwrite a text file in your workspace.

    Path is relative to workspace root. Provide the full file content.
    Creates the file if it doesn't exist, overwrites if it does.
    """
    try:
        written = await run_context.session_state['runtime'].write_file(agent_id=run_context.session_state['agent_id'], path=path, content=content)
    except Exception as exc:
        return f'Error: {exc}'
    return f'Wrote file: {written}'


async def delete_workspace_path(run_context: RunContext, path: str) -> str:
    """Delete a file or directory from your workspace.

    Path is relative to workspace root. Directories are removed recursively.
    """
    try:
        deleted = await run_context.session_state['runtime'].delete_path(agent_id=run_context.session_state['agent_id'], path=path)
    except Exception as exc:
        return f'Error: {exc}'
    if not deleted:
        return f'Path not found: {path}'
    return f'Deleted: {path}'


async def create_scheduled_job(
    run_context: RunContext,
    description: str,
    instruction: str,
    run_at: str | None = None,
    cron_expr: str | None = None,
) -> str:
    """Schedule a task to run in the future.

    Provide exactly one of:
    - run_at: ISO 8601 datetime in UTC (e.g. '2025-01-15T09:00:00') for a one-time job.
    - cron_expr: cron expression (e.g. '0 9 * * *' for daily at 9am UTC) for recurring jobs.

    The instruction is the prompt you will receive when the job triggers.
    Description is a short human-readable label for identification.
    """
    agent_id = run_context.session_state['agent_id']

    if bool(run_at) == bool(cron_expr):
        return 'Error: provide exactly one of run_at (ISO datetime) or cron_expr (cron expression).'

    ts_now = now()

    if run_at:
        try:
            dt = datetime.fromisoformat(run_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            next_run_at = int(dt.timestamp())
        except ValueError:
            return f'Error: invalid ISO datetime: {run_at}'
        if next_run_at <= ts_now:
            return 'Error: run_at must be in the future.'
        job_type = 'once'
    else:
        if not croniter.is_valid(cron_expr):
            return f'Error: invalid cron expression: {cron_expr}'
        dt_now = datetime.fromtimestamp(ts_now, tz=timezone.utc)
        cron = croniter(cron_expr, dt_now)
        next_dt = cron.get_next(datetime)
        next_run_at = int(next_dt.timestamp())
        job_type = 'recurring'

    job_id = await storage.create_scheduled_job(
        agent_id=agent_id,
        description=description.strip(),
        instruction=instruction.strip(),
        job_type=job_type,
        cron_expr=cron_expr,
        next_run_at=next_run_at,
        created_at=ts_now,
    )
    return f'Scheduled job created: id={job_id}, type={job_type}, next_run_at={next_run_at}'


async def list_scheduled_jobs(run_context: RunContext) -> str:
    """List all your active scheduled jobs.

    Shows each job's ID, type (once/recurring), cron expression (if recurring),
    next run time, description, and instruction.
    Use this to review what's already scheduled before creating new jobs.
    """
    agent_id = run_context.session_state['agent_id']
    jobs = await storage.list_scheduled_jobs(agent_id=agent_id)
    if not jobs:
        return 'No active scheduled jobs.'

    lines = []
    for j in jobs:
        cron_part = f' cron={j.cron_expr}' if j.cron_expr else ''
        lines.append(
            f'- id={j.id} | {j.job_type}{cron_part} | next_run={j.next_run_at}\n'
            f'  description={j.description}\n'
            f'  instruction={j.instruction}'
        )
    return 'Scheduled jobs:\n' + '\n'.join(lines)


async def delete_scheduled_job(run_context: RunContext, job_id: str) -> str:
    """Cancel and permanently delete a scheduled job by its ID.

    Use list_scheduled_jobs first to find the job ID.
    """
    agent_id = run_context.session_state['agent_id']
    deleted = await storage.delete_scheduled_job(agent_id=agent_id, job_id=job_id)
    if not deleted:
        return f"Job '{job_id}' not found."
    return f"Job '{job_id}' deleted."


async def send_message(
    run_context: RunContext,
    channel: str,
    content: str,
    chat_id: str | None = None,
) -> str:
    """Send a message to a configured channel (e.g. 'telegram', 'web').

    Use this to proactively reach out, send notifications, or respond on a specific channel.
    If chat_id is omitted, the first configured chat_id for that channel is used as default.
    """
    agent_id = run_context.session_state['agent_id']

    channel_record = await storage.get_agent_channel(agent_id=agent_id, channel=channel)
    if channel_record is None or not channel_record.enabled:
        return f"Error: channel '{channel}' is not configured or not enabled for this agent."

    resolved_chat_id = chat_id
    if not resolved_chat_id:
        # Try to resolve from channel config
        cfg = channel_record.config
        allowed = cfg.get('allowed_chat_ids')
        if isinstance(allowed, list) and allowed:
            resolved_chat_id = str(allowed[0])
        elif channel == 'web':
            resolved_chat_id = 'web-ui'

    if not resolved_chat_id:
        return f"Error: no chat_id provided and could not resolve a default for channel '{channel}'."

    if 'pending_outbound' not in run_context.session_state:
        run_context.session_state['pending_outbound'] = []
    pending: list[OutboundMessage] = run_context.session_state['pending_outbound']
    pending.append(
        OutboundMessage(
            agent_id=agent_id,
            channel=channel,
            chat_id=resolved_chat_id,
            content=content,
        )
    )
    return f'Message queued for channel={channel}, chat_id={resolved_chat_id}.'


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
    create_scheduled_job,
    list_scheduled_jobs,
    delete_scheduled_job,
    send_message,
]
