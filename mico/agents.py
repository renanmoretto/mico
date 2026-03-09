from __future__ import annotations

from . import agent_config
from . import storage
from .utils import now, slugify


async def create_agent(*, name: str) -> storage.AgentRecord:
    agent_name = slugify(name)
    created_at = now()
    agent_id = await storage.create_agent(
        name=agent_name,
        created_at=created_at,
    )
    row = await storage.get_agent(agent_id)
    if row is None:
        raise RuntimeError(f'Failed to create agent {agent_name}.')
    await agent_config.ensure_agent_config(agent_id)
    return row


async def ensure_agent(*, name: str = 'default') -> storage.AgentRecord:
    existing = await storage.get_agent_by_name(name)
    if existing is not None:
        return existing
    return await create_agent(name=name)


async def require_agent(agent_id: str) -> storage.AgentRecord:
    row = await storage.get_agent(agent_id)
    if row is None:
        raise ValueError(f"Agent '{agent_id}' not found.")
    return row


async def list_agents() -> list[storage.AgentRecord]:
    return await storage.list_agents()


async def delete_agent(agent_id: str) -> bool:
    return await storage.delete_agent(agent_id)


async def configure_channel(
    *,
    agent_id: str,
    channel: str,
    enabled: bool,
    config: dict[str, object] | None = None,
) -> agent_config.AgentChannel:
    return await agent_config.configure_channel(
        agent_id=agent_id,
        channel=channel,
        enabled=enabled,
        config=dict(config or {}),
    )
