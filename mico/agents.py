from __future__ import annotations

from . import storage
from .utils import now, slugify


def create_agent(*, name: str, persona: str = '') -> storage.AgentRecord:
    agent_name = slugify(name)
    created_at = now()
    agent_id = storage.create_agent(
        name=agent_name,
        persona=persona.strip(),
        created_at=created_at,
    )
    row = storage.get_agent(agent_id)
    if row is None:
        raise RuntimeError(f'Failed to create agent {agent_name}.')
    return row


def ensure_agent(*, name: str = 'default', persona: str = '') -> storage.AgentRecord:
    existing = storage.find_agent(name)
    if existing is not None:
        return existing
    return create_agent(name=name, persona=persona)


def require_agent(agent_id: str) -> storage.AgentRecord:
    row = storage.get_agent(agent_id)
    if row is None:
        raise ValueError(f"Agent '{agent_id}' not found.")
    return row


def list_agents() -> list[storage.AgentRecord]:
    return storage.list_agents()


def update_agent_persona(*, agent_id: str, persona: str) -> storage.AgentRecord:
    if not storage.update_agent(agent_id=agent_id, persona=persona, updated_at=now()):
        raise ValueError(f"Agent '{agent_id}' not found.")
    return storage.get_agent(agent_id)  # type: ignore[return-value]


def delete_agent(agent_id: str) -> bool:
    return storage.delete_agent(agent_id)


def configure_channel(
    *,
    agent_id: str,
    channel: str,
    enabled: bool,
    config: dict[str, object] | None = None,
) -> storage.AgentChannelRecord:
    storage.upsert_agent_channel(
        agent_id=agent_id,
        channel=channel,
        enabled=enabled,
        updated_at=now(),
        config=dict(config or {}),
    )
    row = storage.get_agent_channel(agent_id=agent_id, channel=channel)
    if row is None:
        raise RuntimeError(f'Failed to configure {channel} for agent {agent_id}.')
    return row
