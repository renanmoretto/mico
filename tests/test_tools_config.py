from __future__ import annotations

import asyncio
import json

from agno.run import RunContext

from mico import agents, runtime, storage, tools


def test_update_config_tool_validates_and_saves(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'tools-config.db'
        base_dir = tmp_path / 'agents'
        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='tool-config')
        run_context = RunContext(
            run_id='run-1',
            session_id='session-1',
            session_state={'agent_id': row.id, 'runtime': manager},
        )

        current = await tools.get_config(run_context)
        payload = json.loads(current)
        payload['channels']['telegram'] = {
            'enabled': True,
            'bot_token': 'tool-secret',
            'allowed_chat_ids': ['111'],
        }

        saved = await tools.update_config(run_context, json.dumps(payload))

        assert 'Saved config:' in saved
        latest = json.loads(await tools.get_config(run_context))
        assert latest['channels']['telegram']['enabled'] is True
        assert latest['channels']['telegram']['bot_token'] == 'tool-secret'
        assert latest['channels']['telegram']['allowed_chat_ids'] == ['111']

    asyncio.run(scenario())


def test_update_config_tool_rejects_invalid_payload_without_overwriting(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'tools-config-invalid.db'
        base_dir = tmp_path / 'agents'
        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='tool-config-invalid')
        run_context = RunContext(
            run_id='run-2',
            session_id='session-2',
            session_state={'agent_id': row.id, 'runtime': manager},
        )

        before = json.loads(await tools.get_config(run_context))
        result = await tools.update_config(
            run_context,
            json.dumps({'channels': {'web': {'enabled': True}}}),
        )
        after = json.loads(await tools.get_config(run_context))

        assert result.startswith('Error: Invalid agent config:')
        assert after == before

    asyncio.run(scenario())


def test_workspace_tools_block_config_json_mutation(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'tools-config-guard.db'
        base_dir = tmp_path / 'agents'
        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='tool-config-guard')
        run_context = RunContext(
            run_id='run-3',
            session_id='session-3',
            session_state={'agent_id': row.id, 'runtime': manager},
        )

        write_result = await tools.write_workspace_file(run_context, './config.json', '{}')
        delete_result = await tools.delete_workspace_path(run_context, 'config.json')

        assert write_result == 'Error: config.json is protected. Use get_config and update_config instead.'
        assert delete_result == 'Error: config.json is protected. Use update_config instead.'

    asyncio.run(scenario())


def test_attached_roots_are_not_exposed_through_agent_config_tools(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'tools-attached-roots.db'
        base_dir = tmp_path / 'agents'
        shared = tmp_path / 'shared-tools'
        shared.mkdir()

        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='tools-attached-roots')
        await manager.attach_folder(agent_id=row.id, path=str(shared), name='docs')
        run_context = RunContext(
            run_id='run-4',
            session_id='session-4',
            session_state={'agent_id': row.id, 'runtime': manager},
        )

        current = json.loads(await tools.get_config(run_context))
        assert 'attached_folders' not in current

        patched = dict(current)
        patched['attached_folders'] = [{'name': 'hack', 'path': str(tmp_path)}]
        result = await tools.update_config(run_context, json.dumps(patched))
        assert result.startswith('Error: Invalid agent config:')

        listed = await tools.list_workspace_files(run_context, root='docs')
        write_result = await tools.write_workspace_file(run_context, 'config.json', '{"ok":true}', root='docs')
        read_result = await tools.read_workspace_file(run_context, 'config.json', root='docs')
        delete_result = await tools.delete_workspace_path(run_context, 'config.json', root='docs')

        assert listed == 'No files found.'
        assert write_result == 'Wrote file: config.json'
        assert read_result == '{"ok":true}'
        assert delete_result == 'Deleted: config.json'
        assert not (shared / 'config.json').exists()
        assert [folder.name for folder in await manager.list_attached_folders(row.id)] == ['docs']

    asyncio.run(scenario())


def test_list_workspace_roots_returns_current_root_names(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'tools-roots.db'
        base_dir = tmp_path / 'agents'
        shared = tmp_path / 'shared-roots'
        shared.mkdir()

        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='tools-roots')
        await manager.attach_folder(agent_id=row.id, path=str(shared), name='docs')
        run_context = RunContext(
            run_id='run-5',
            session_id='session-5',
            session_state={'agent_id': row.id, 'runtime': manager},
        )

        result = await tools.list_workspace_roots(run_context)

        assert result == 'Roots:\n- workspace (default)\n- docs'

    asyncio.run(scenario())
