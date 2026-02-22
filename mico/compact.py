import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from . import storage
from .utils import now

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency fallback
    tiktoken = None


DEFAULT_TOKEN_MODEL = 'google/gemini-2.5-flash-lite'


@dataclass(frozen=True)
class CompactionResult:
    triggered: bool
    total_tokens_before: int
    total_tokens_after: int
    compacted_messages: int
    compacted_tokens: int
    kept_recent_messages: int
    kept_recent_tokens: int
    memory_name: str | None = None


@lru_cache(maxsize=8)
def _get_encoder(model: str):
    if tiktoken is None:
        return None

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding('cl100k_base')


def count_tokens(text: str, model: str = DEFAULT_TOKEN_MODEL) -> int:
    if not text:
        return 0

    encoder = _get_encoder(model=model)
    if encoder is None:
        return max(1, math.ceil(len(text) / 4))
    return len(encoder.encode(text))


def message_tokens(role: str, content: str, model: str = DEFAULT_TOKEN_MODEL) -> int:
    return 4 + count_tokens(role, model=model) + count_tokens(content, model=model)


def select_recent_messages_for_context(
    *,
    agent_id: str,
    token_budget: int = 20_000,
    model: str = DEFAULT_TOKEN_MODEL,
) -> list[storage.MessageRecord]:
    rows = storage.list_messages_with_order(agent_id=agent_id)
    if not rows:
        return []

    selected: list[storage.MessageRecord] = []
    total = 0
    for row in reversed(rows):
        tokens = message_tokens(str(row.role), str(row.content or ''), model=model)
        if selected and total + tokens > token_budget:
            break
        selected.append(row)
        total += tokens
    selected.reverse()
    return selected


def _find_recent_tail_start(token_counts: list[int], keep_recent_tokens: int) -> tuple[int, int]:
    if not token_counts:
        return 0, 0

    tail_tokens = 0
    tail_start = len(token_counts)
    for index in range(len(token_counts) - 1, -1, -1):
        next_total = tail_tokens + token_counts[index]
        if tail_start < len(token_counts) and next_total > keep_recent_tokens:
            break
        tail_tokens = next_total
        tail_start = index
    return tail_start, tail_tokens


def _build_compaction_memory_payload(
    *,
    agent_id: str,
    compacted_rows: list[storage.MessageRecord],
    compacted_tokens: int,
    total_before: int,
    total_after: int,
    keep_recent_tokens: int,
    model: str,
) -> tuple[str, str]:
    first = compacted_rows[0]
    last = compacted_rows[-1]
    created_at = now()
    memory_name = f'conversation_compaction_{created_at}_{first.id[:8]}'
    summary = (
        f'Auto-compacted {len(compacted_rows)} old messages '
        f'(~{compacted_tokens} tokens) to keep active context focused on recent turns.'
    )

    payload: dict[str, Any] = {
        'type': 'conversation_compaction',
        'agent_id': agent_id,
        'created_at': created_at,
        'token_model': model,
        'stats': {
            'compacted_messages': len(compacted_rows),
            'compacted_tokens': compacted_tokens,
            'total_tokens_before': total_before,
            'total_tokens_after': total_after,
            'recent_context_target_tokens': keep_recent_tokens,
        },
        'range': {
            'first_message_id': first.id,
            'last_message_id': last.id,
            'first_timestamp': first.timestamp,
            'last_timestamp': last.timestamp,
        },
        'messages': [
            {
                'id': row.id,
                'timestamp': row.timestamp,
                'role': row.role,
                'content': row.content,
            }
            for row in compacted_rows
        ],
    }
    return memory_name, summary + '\n\n' + json.dumps(payload, ensure_ascii=True)


def compact_conversation_if_needed(
    *,
    agent_id: str,
    threshold_tokens: int = 100_000,
    target_tokens_after: int = 80_000,
    keep_recent_tokens: int = 20_000,
    model: str = DEFAULT_TOKEN_MODEL,
    memory_strength: int = 4,
) -> CompactionResult:
    if threshold_tokens <= 0 or target_tokens_after <= 0 or keep_recent_tokens <= 0:
        raise ValueError('Token thresholds must be positive.')
    if target_tokens_after > threshold_tokens:
        raise ValueError('target_tokens_after must be <= threshold_tokens.')

    rows = storage.list_messages_with_order(agent_id=agent_id)
    if not rows:
        return CompactionResult(False, 0, 0, 0, 0, 0, 0, None)

    token_counts = [message_tokens(str(row.role), str(row.content or ''), model=model) for row in rows]
    total_before = sum(token_counts)
    if total_before < threshold_tokens:
        tail_start, tail_tokens = _find_recent_tail_start(token_counts, keep_recent_tokens=keep_recent_tokens)
        return CompactionResult(
            False,
            total_before,
            total_before,
            0,
            0,
            len(rows[tail_start:]),
            tail_tokens,
            None,
        )

    tail_start, tail_tokens = _find_recent_tail_start(token_counts, keep_recent_tokens=keep_recent_tokens)
    tokens_to_remove = max(0, total_before - target_tokens_after)

    remove_end = 0
    compacted_tokens = 0
    while remove_end < tail_start and compacted_tokens < tokens_to_remove:
        compacted_tokens += token_counts[remove_end]
        remove_end += 1

    compacted_rows = rows[:remove_end]
    kept_rows = rows[remove_end:]
    total_after = total_before - compacted_tokens
    kept_recent_messages = len(kept_rows)

    if not compacted_rows:
        return CompactionResult(
            False,
            total_before,
            total_after,
            0,
            0,
            kept_recent_messages,
            sum(token_counts[remove_end:]),
            None,
        )

    memory_name, memory_content = _build_compaction_memory_payload(
        agent_id=agent_id,
        compacted_rows=compacted_rows,
        compacted_tokens=compacted_tokens,
        total_before=total_before,
        total_after=total_after,
        keep_recent_tokens=keep_recent_tokens,
        model=model,
    )
    storage.upsert_memory(
        agent_id=agent_id,
        name=memory_name.strip(),
        summary=(
            f'Conversation history compacted at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now()))}.'
        ).strip(),
        content=memory_content.strip(),
        strength=memory_strength,
        updated_at=now(),
    )
    storage.delete_messages(agent_id=agent_id, message_ids=[row.id for row in compacted_rows])

    return CompactionResult(
        True,
        total_before,
        total_after,
        len(compacted_rows),
        compacted_tokens,
        kept_recent_messages,
        sum(token_counts[remove_end:]),
        memory_name,
    )
