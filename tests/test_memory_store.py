from __future__ import annotations

import asyncio
import re
import tomllib

from mico import memory_store
from mico.runtime import RuntimeManager


def test_memory_store_writes_markdown_files(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        manager = RuntimeManager(base_dir=str(tmp_path / 'agents'), docker_enabled=False)
        monkeypatch.setattr(memory_store, 'get_runtime_manager', lambda: manager)

        await memory_store.upsert_memory(
            agent_id='agent-1',
            name='User Preferences',
            summary='Concise answers',
            content='Keep replies short.',
            strength=5,
        )

        files = sorted((manager.workspace_path('agent-1') / 'memories').glob('*.md'))
        assert len(files) == 1
        text = files[0].read_text(encoding='utf-8')
        assert re.search(r'^created_at = \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', text, re.MULTILINE)
        assert re.search(r'^updated_at = \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', text, re.MULTILINE)
        front_matter = tomllib.loads(text.split('+++\n', 2)[1])
        assert front_matter['name'] == 'User Preferences'
        assert text.rstrip().endswith('Keep replies short.')

    asyncio.run(scenario())


def test_memory_store_search_update_and_delete(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        manager = RuntimeManager(base_dir=str(tmp_path / 'agents'), docker_enabled=False)
        monkeypatch.setattr(memory_store, 'get_runtime_manager', lambda: manager)

        await memory_store.upsert_memory(
            agent_id='agent-1',
            name='Preference',
            summary='First version',
            content='Short answers.',
            strength=3,
        )
        row = await memory_store.find_memory(agent_id='agent-1', identifier='Preference')
        assert row is not None

        await memory_store.update_memory(
            agent_id='agent-1',
            memory_id=row.id,
            name='Preference',
            summary='Updated',
            content='Short, direct answers.',
            strength=4,
        )
        rows = await memory_store.search_memories(agent_id='agent-1', query='direct', limit=10)
        assert [item.summary for item in rows] == ['Updated']
        assert rows[0].updated_at.endswith('Z')

        await memory_store.delete_memory(agent_id='agent-1', memory_id=row.id)
        assert await memory_store.find_memory(agent_id='agent-1', identifier=row.id) is None

    asyncio.run(scenario())
