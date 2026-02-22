from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_web_config_page_loads(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-config.db'
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.web as web

    web = importlib.reload(web)
    with TestClient(web.app) as client:
        page = client.get('/config')
        assert page.status_code == 200
        assert 'Application Config' in page.text
        assert 'Config JSON' in page.text


def test_web_config_rejects_invalid_json(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-config-invalid.db'
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.web as web

    web = importlib.reload(web)
    with TestClient(web.app) as client:
        resp = client.post(
            '/config',
            data={'config_json': '{invalid'},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        location = resp.headers.get('location', '')
        assert location.startswith('/config?')
        assert 'error=' in location


def test_web_config_saves_json(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-config-save.db'
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.web as web

    web = importlib.reload(web)
    with TestClient(web.app) as client:
        resp = client.post(
            '/config',
            data={
                'config_json': (
                    '{"model":{"openrouter_model":"openai/gpt-4o-mini"},'
                    '"runtime":{"docker_enabled":false}}'
                )
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303

        stored = web.storage.get_config(key='app')
        assert stored is not None
        assert stored.get('model', {}).get('openrouter_model') == 'openai/gpt-4o-mini'
        assert stored.get('runtime', {}).get('docker_enabled') is False

        page = client.get('/config')
        assert page.status_code == 200
        assert 'openai/gpt-4o-mini' in page.text
