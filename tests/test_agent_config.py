from __future__ import annotations

import asyncio
import json

from mico import agent_config, agents, runtime, storage


def test_create_agent_seeds_workspace_config(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'agent-config-seed.db'
        base_dir = tmp_path / 'agents'
        await storage.init_storage(db_path=str(db_path))
        monkeypatch.setattr(
            runtime,
            '_RUNTIME_MANAGER',
            runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False),
        )

        row = await agents.create_agent(name='config-seed')

        config_path = base_dir / row.id / 'workspace' / 'config.json'
        assert config_path.exists()

        payload = json.loads(config_path.read_text(encoding='utf-8'))
        assert payload['llm'] is None
        assert payload['channels']['telegram']['enabled'] is False
        assert payload['channels']['telegram']['bot_token'] == ''
        assert payload['channels']['telegram']['allowed_chat_ids'] == []

    asyncio.run(scenario())


def test_configure_channel_writes_workspace_config(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'agent-config-write.db'
        base_dir = tmp_path / 'agents'
        await storage.init_storage(db_path=str(db_path))
        monkeypatch.setattr(
            runtime,
            '_RUNTIME_MANAGER',
            runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False),
        )

        row = await agents.create_agent(name='telegram-agent')
        updated = await agent_config.configure_channel(
            agent_id=row.id,
            channel='telegram',
            enabled=True,
            config={
                'bot_token': 'secret-token',
                'allowed_chat_ids': ['123', '456'],
            },
        )

        assert updated.enabled is True
        assert updated.config['bot_token'] == 'secret-token'
        assert updated.config['allowed_chat_ids'] == ['123', '456']

        payload = await agent_config.get_agent_config(row.id)
        assert payload['channels']['telegram']['enabled'] is True
        assert payload['channels']['telegram']['bot_token'] == 'secret-token'
        assert payload['channels']['telegram']['allowed_chat_ids'] == ['123', '456']

        listed = await agent_config.list_agent_channels(row.id)
        assert [item.channel for item in listed] == ['telegram']

    asyncio.run(scenario())


def test_agent_llm_override_falls_back_to_global_when_null(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'agent-config-llm.db'
        base_dir = tmp_path / 'agents'
        await storage.init_storage(db_path=str(db_path))
        monkeypatch.setattr(
            runtime,
            '_RUNTIME_MANAGER',
            runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False),
        )

        await storage.upsert_config(
            key='app',
            config={'llm': {'provider': 'openrouter', 'model': 'openai/gpt-4o-mini'}},
        )
        row = await agents.create_agent(name='llm-agent')

        inherited = await agent_config.get_agent_llm_config(row.id)
        assert inherited.provider == 'openrouter'
        assert inherited.model == 'openai/gpt-4o-mini'

        await agent_config.set_agent_config(
            row.id,
            {
                'llm': {'provider': 'openrouter', 'model': 'google/gemini-3-flash-preview'},
                'channels': {'telegram': {'enabled': False}},
            },
        )
        overridden = await agent_config.get_agent_llm_config(row.id)
        assert overridden.provider == 'openrouter'
        assert overridden.model == 'google/gemini-3-flash-preview'

        await agent_config.set_agent_config(
            row.id,
            {
                'llm': None,
                'channels': {'telegram': {'enabled': False}},
            },
        )
        reset = await agent_config.get_agent_llm_config(row.id)
        assert reset.provider == 'openrouter'
        assert reset.model == 'openai/gpt-4o-mini'

    asyncio.run(scenario())
