import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from . import storage

if TYPE_CHECKING:
    from .agent import Mico


def _now() -> int:
    return int(time.time())


def _clamp(limit: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(limit, maximum))


def _format_ts(ts: int) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))


def _parse_human_time(when: str) -> int | None:
    text = when.strip()
    if not text:
        return None

    if text.isdigit():
        raw = int(text)
        return raw // 1000 if raw > 10_000_000_000 else raw

    iso = text.replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(iso)
        return int(parsed.timestamp())
    except ValueError:
        pass

    now_dt = datetime.now().astimezone()
    lower = text.lower()

    relative = re.match(
        r'^in\s+(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)$',
        lower,
    )
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        if unit.startswith('second'):
            delta = timedelta(seconds=amount)
        elif unit.startswith('minute'):
            delta = timedelta(minutes=amount)
        elif unit.startswith('hour'):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return int((now_dt + delta).timestamp())

    day_phrase = re.match(
        r'^(today|tomorrow)(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$',
        lower,
    )
    if day_phrase:
        day_word = day_phrase.group(1)
        hour = int(day_phrase.group(2))
        minute = int(day_phrase.group(3) or '0')
        meridiem = day_phrase.group(4)

        if meridiem:
            if hour == 12:
                hour = 0
            if meridiem == 'pm':
                hour += 12
        if hour > 23 or minute > 59:
            return None

        target = now_dt
        if day_word == 'tomorrow':
            target = target + timedelta(days=1)
        target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return int(target.timestamp())

    return None


def create_cron(name: str, prompt: str, when: str) -> str:
    parsed = _parse_human_time(when=when)
    if parsed is None:
        return (
            "Error: couldn't parse when. Use ISO datetime (e.g. 2026-03-01T15:00:00), "
            "unix timestamp, 'tomorrow 3pm', or 'in 2 hours'."
        )
    now = _now()
    if parsed <= now:
        return 'Error: cron time must be in the future.'

    cron_name = name.strip() if name.strip() else prompt.strip()[:40] or 'scheduled_task'
    cron_id = storage.create_cron(
        name=cron_name,
        prompt=prompt.strip(),
        run_at=parsed,
        created_at=now,
    )
    return f'Cron scheduled: id={cron_id} | at={_format_ts(parsed)} | name={cron_name}'


def list_crons(limit: int = 20, include_done: bool = False) -> str:
    rows = storage.list_crons(limit=_clamp(limit, 1, 100), include_done=include_done)
    if not rows:
        return 'No cron jobs found.'

    lines = []
    for row in rows:
        line = (
            f'- id={row.id} | status={row.status} | at={_format_ts(row.run_at)} | '
            f'name={row.name} | prompt={row.prompt}'
        )
        if row.last_error:
            line += f'\n  error={row.last_error}'
        lines.append(line)
    return 'Crons:\n' + '\n'.join(lines)


def delete_cron(identifier: str) -> str:
    if storage.delete_cron(cron_id=identifier):
        return f'Cron {identifier} deleted.'
    return f'Cron {identifier} not found.'


def _build_cron_system_message(row: storage.CronRecord) -> str:
    return (
        'You are handling an automated cron trigger. '
        f'Cron id={row.id}, name={row.name}, scheduled_for={_format_ts(row.run_at)}. '
        'Treat the next user message as the reminder/task that must be executed now.'
    )


async def process_due_crons(mico: 'Mico', max_jobs: int = 10) -> int:
    now = _now()
    due = storage.claim_due_crons(now=now, limit=_clamp(max_jobs, 1, 100))
    if not due:
        return 0

    for row in due:
        try:
            await mico.run(
                user_input=row.prompt,
                system_input=_build_cron_system_message(row),
                quiet=True,
            )
            storage.mark_cron_done(cron_id=row.id, finished_at=_now())
        except Exception as exc:
            storage.mark_cron_error(cron_id=row.id, error=str(exc), finished_at=_now())
    return len(due)


async def run_worker(mico: 'Mico', poll_seconds: float = 2.0, once: bool = False) -> None:
    poll = max(0.2, poll_seconds)
    while True:
        processed = await process_due_crons(mico=mico)
        if once:
            return
        await asyncio.sleep(0 if processed else poll)
