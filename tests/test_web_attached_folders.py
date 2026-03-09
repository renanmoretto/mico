from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_web_folder_picker_api_returns_selected_path(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-folder-picker.db'
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.web as web

    web = importlib.reload(web)

    async def fake_pick_folder() -> str | None:
        return str((tmp_path / 'picked').resolve())

    monkeypatch.setattr(web, '_pick_folder', fake_pick_folder)

    with TestClient(web.app) as client:
        resp = client.post('/api/folders/pick')
        assert resp.status_code == 200
        assert resp.json() == {'path': str((tmp_path / 'picked').resolve())}


def test_web_attached_folders_attach_detach_and_delete_keep_source(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-attached-folders.db'
    base_dir = tmp_path / 'agents'
    shared = tmp_path / 'shared-web'
    shared.mkdir()
    (shared / 'notes.txt').write_text('hello', encoding='utf-8')
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.runtime as runtime
    import mico.web as web

    web = importlib.reload(web)
    monkeypatch.setattr(
        runtime,
        '_RUNTIME_MANAGER',
        runtime.RuntimeManager(base_dir=str(base_dir), docker_enabled=False),
    )

    with TestClient(web.app) as client:
        created = client.post('/agents', data={'name': 'web-attached'}, follow_redirects=False)
        assert created.status_code == 303
        agent_id = created.headers['location'].split('/agents/')[1].split('?')[0]

        attached = client.post(
            f'/agents/{agent_id}/folders/attach',
            data={'path': str(shared), 'name': 'docs'},
            follow_redirects=False,
        )
        assert attached.status_code == 303
        assert 'root=docs' in attached.headers['location']

        page = client.get(f'/agents/{agent_id}?tab=workspace&root=docs&path=.')
        assert page.status_code == 200
        assert 'roots' in page.text
        assert 'attach folder' in page.text
        assert 'notes.txt' in page.text
        assert str(shared) in page.text

        detached = client.post(
            f'/agents/{agent_id}/folders/detach',
            data={'name': 'docs'},
            follow_redirects=False,
        )
        assert detached.status_code == 303
        assert shared.exists()
        assert (shared / 'notes.txt').exists()

        reattached = client.post(
            f'/agents/{agent_id}/folders/attach',
            data={'path': str(shared), 'name': 'docs'},
            follow_redirects=False,
        )
        assert reattached.status_code == 303

        deleted = client.post(f'/agents/{agent_id}/delete', follow_redirects=False)
        assert deleted.status_code == 303

    assert shared.exists()
    assert (shared / 'notes.txt').read_text(encoding='utf-8') == 'hello'
