from __future__ import annotations

import importlib
import json

from fastapi.testclient import TestClient


def test_web_telegram_settings_save_to_workspace_config(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-agent-config.db'
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

    async def noop_reload() -> None:
        return None

    monkeypatch.setattr(web, '_reload_telegram_service', noop_reload)

    with TestClient(web.app) as client:
        created = client.post('/agents', data={'name': 'web-telegram'}, follow_redirects=False)
        assert created.status_code == 303
        agent_id = created.headers['location'].split('/agents/')[1].split('?')[0]

        resp = client.post(
            f'/agents/{agent_id}/telegram',
            data={
                'enabled': 'on',
                'bot_token': 'web-secret',
                'allowed_chat_ids': '123\n456',
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303

    config_path = base_dir / agent_id / 'workspace' / 'config.json'
    payload = json.loads(config_path.read_text(encoding='utf-8'))
    assert payload['channels']['telegram']['enabled'] is True
    assert payload['channels']['telegram']['bot_token'] == 'web-secret'
    assert payload['channels']['telegram']['allowed_chat_ids'] == ['123', '456']


def test_web_agent_routes_require_agent_id(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-agent-id-only.db'
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

    with TestClient(web.app) as client:
        created = client.post('/agents', data={'name': 'id-only'}, follow_redirects=False)
        assert created.status_code == 303

        resp = client.get('/agents/id-only')
        assert resp.status_code == 404
