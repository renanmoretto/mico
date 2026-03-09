from __future__ import annotations

import asyncio
import importlib

from fastapi.testclient import TestClient


def test_mico_build_history_replays_first_class_tool_fields(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'message-schema.db'
        base_dir = tmp_path / 'agents'

        import mico.agents as agents
        import mico.runtime as runtime
        import mico.storage as storage
        from mico.agent import Mico

        await storage.init_storage(db_path=str(db_path))
        runtime.reset_runtime_manager()
        runtime._RUNTIME_MANAGER = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)

        row = await agents.create_agent(name='schema-agent')
        tool_calls = [
            {
                'id': 'call_1',
                'type': 'function',
                'function': {
                    'name': 'search_messages',
                    'arguments': '{"query":"hello"}',
                },
            }
        ]
        await storage.add_message(agent_id=row.id, role='user', content='hello', timestamp=1)
        await storage.add_message(agent_id=row.id, role='assistant', content=None, timestamp=2, tool_calls=tool_calls)
        await storage.add_message(
            agent_id=row.id,
            role='tool',
            content='1 hit',
            timestamp=3,
            tool_call_id='call_1',
            metadata={'tool_name': 'search_messages', 'tool_call_error': True},
        )

        history = await Mico()._build_history_input(agent_id=row.id, token_budget=10_000)

        assert [msg.role for msg in history] == ['user', 'assistant', 'tool']
        assert history[1].tool_calls == tool_calls
        assert history[2].tool_call_id == 'call_1'
        assert history[2].name is None
        assert history[2].tool_name == 'search_messages'
        assert history[2].tool_call_error is True

    asyncio.run(scenario())


def test_web_messages_use_first_class_tool_fields(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-message-schema.db'
    base_dir = tmp_path / 'agents'
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.runtime as runtime
    import mico.web as web

    web = importlib.reload(web)
    monkeypatch.setattr(
        runtime,
        '_RUNTIME_MANAGER',
        runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False),
    )

    async def seed() -> str:
        await web._ensure_storage()
        row = await web.agents.create_agent(name='web-message-agent')
        tool_calls = [
            {
                'id': 'call_1',
                'type': 'function',
                'function': {
                    'name': 'search_messages',
                    'arguments': '{"query":"hello"}',
                },
            }
        ]
        await web.storage.add_message(agent_id=row.id, role='assistant', content=None, timestamp=1, tool_calls=tool_calls)
        await web.storage.add_message(
            agent_id=row.id,
            role='tool',
            content='1 hit',
            timestamp=2,
            tool_call_id='call_1',
            metadata={'tool_name': 'search_messages'},
        )
        await web.storage.add_message(agent_id=row.id, role='assistant', content='done', timestamp=3)
        return row.id

    agent_id = asyncio.run(seed())

    with TestClient(web.app) as client:
        page = client.get(f'/agents/{agent_id}')
        assert page.status_code == 200
        assert 'activity-item done">search_messages' in page.text
        assert 'done' in page.text
        assert '{"query":"hello"}' not in page.text

        payload = client.get(f'/api/agents/{agent_id}/messages')
        assert payload.status_code == 200
        messages = payload.json()['messages']

    assert messages[0]['tool_calls'][0]['id'] == 'call_1'
    assert 'content' not in messages[0]
    assert messages[1]['metadata']['tool_name'] == 'search_messages'
    assert messages[1]['tool_call_id'] == 'call_1'
    assert 'name' not in messages[1]
    assert messages[2]['content'] == 'done'


def test_search_messages_matches_tool_calls(tmp_path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'message-search.db'

        import mico.storage as storage

        await storage.init_storage(db_path=str(db_path))
        agent_id = await storage.create_agent(name='search-agent', created_at=1)
        await storage.add_message(
            agent_id=agent_id,
            role='assistant',
            content=None,
            timestamp=1,
            tool_calls=[
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'search_messages', 'arguments': '{"query":"hello"}'},
                }
            ],
        )

        rows = await storage.search_messages(agent_id=agent_id, query='search_messages', limit=10)

        assert len(rows) == 1
        assert rows[0].tool_calls is not None
        assert rows[0].tool_calls[0]['id'] == 'call_1'

    asyncio.run(scenario())
