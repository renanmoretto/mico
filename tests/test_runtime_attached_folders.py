from __future__ import annotations

import asyncio

from mico import agents, runtime, storage


def test_attach_folder_persists_in_agent_metadata(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'runtime-attached.db'
        base_dir = tmp_path / 'agents'
        shared = tmp_path / 'shared-docs'
        shared.mkdir()

        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='attached-runtime')
        folder = await manager.attach_folder(agent_id=row.id, path=str(shared), name='Docs Root')

        assert folder == runtime.AttachedFolder(name='docs-root', path=str(shared.resolve()))

        saved = await storage.get_agent(row.id)
        assert saved is not None
        assert saved.metadata == {
            'attached_folders': [{'name': 'docs-root', 'path': str(shared.resolve())}],
        }

    asyncio.run(scenario())


def test_attach_folder_rejects_invalid_input(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'runtime-attached-invalid.db'
        base_dir = tmp_path / 'agents'
        shared = tmp_path / 'shared'
        shared.mkdir()
        duplicate = tmp_path / 'duplicate'
        duplicate.mkdir()
        inside_runtime = base_dir / 'nested'
        inside_runtime.mkdir(parents=True)

        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='attached-invalid')
        await manager.attach_folder(agent_id=row.id, path=str(shared), name='docs')

        cases = [
            ({'path': 'relative/path', 'name': 'rel'}, 'must be absolute'),
            ({'path': str(tmp_path / 'missing'), 'name': 'missing'}, 'does not exist'),
            ({'path': str(duplicate), 'name': 'docs'}, 'already exists'),
            ({'path': str(duplicate), 'name': 'workspace'}, 'reserved'),
            ({'path': str(inside_runtime), 'name': 'inside'}, 'managed runtime directory'),
        ]

        for payload, expected in cases:
            try:
                await manager.attach_folder(agent_id=row.id, **payload)
            except ValueError as exc:
                assert expected in str(exc)
            else:
                assert False, f'Expected ValueError containing {expected!r}'

    asyncio.run(scenario())


def test_attached_root_file_ops_block_escape_and_detach_keeps_files(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'runtime-attached-files.db'
        base_dir = tmp_path / 'agents'
        shared = tmp_path / 'shared-files'
        shared.mkdir()
        (shared / 'notes.txt').write_text('hello', encoding='utf-8')

        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='attached-files')
        await manager.attach_folder(agent_id=row.id, path=str(shared), name='docs')

        assert await manager.list_files(agent_id=row.id, root='docs', path='.') == ['notes.txt']
        assert await manager.read_file(agent_id=row.id, root='docs', path='notes.txt') == 'hello'

        written = await manager.write_file(agent_id=row.id, root='docs', path='nested/out.txt', content='world')
        assert written == 'nested/out.txt'
        assert (shared / 'nested' / 'out.txt').read_text(encoding='utf-8') == 'world'

        try:
            await manager.read_file(agent_id=row.id, root='docs', path='../secret.txt')
        except ValueError as exc:
            assert 'escapes the selected root' in str(exc)
        else:
            assert False, 'Expected read_file to reject path escapes.'

        detached = await manager.detach_folder(agent_id=row.id, name='docs')
        assert detached is True
        assert shared.exists()
        assert (shared / 'notes.txt').exists()

        try:
            await manager.list_files(agent_id=row.id, root='docs')
        except ValueError as exc:
            assert "Root 'docs' not found." == str(exc)
        else:
            assert False, 'Expected detached root to become unavailable.'

    asyncio.run(scenario())


def test_delete_agent_runtime_keeps_attached_folder_contents(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'runtime-attached-delete.db'
        base_dir = tmp_path / 'agents'
        shared = tmp_path / 'shared-delete'
        shared.mkdir()
        (shared / 'keep.txt').write_text('keep', encoding='utf-8')

        await storage.init_storage(db_path=str(db_path))
        manager = runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False)
        monkeypatch.setattr(runtime, '_RUNTIME_MANAGER', manager)

        row = await agents.create_agent(name='attached-delete')
        await manager.attach_folder(agent_id=row.id, path=str(shared), name='docs')

        await manager.delete_agent_runtime(row.id)
        deleted = await agents.delete_agent(row.id)

        assert deleted is True
        assert shared.exists()
        assert (shared / 'keep.txt').read_text(encoding='utf-8') == 'keep'

    asyncio.run(scenario())
