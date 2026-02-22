from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_web_chat_route_uses_bus_pipeline(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'web-bus.db'
    monkeypatch.setenv('MICO_DB_PATH', str(db_path))

    import mico.web as web

    web = importlib.reload(web)
    calls: list[dict[str, object]] = []

    async def fake_run(
        *,
        agent_id: str,
        user_input: str | None = None,
        system_input: str | None = None,
        quiet: bool = False,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        calls.append(
            {
                'agent_id': agent_id,
                'user_input': user_input,
                'channel': channel,
                'chat_id': chat_id,
                'sender_id': sender_id,
                'metadata': dict(metadata or {}),
            }
        )
        return 'stub-reply'

    monkeypatch.setattr(web._mico, 'run', fake_run)

    with TestClient(web.app) as client:
        created = client.post(
            '/agents',
            data={'name': 'web-bus-agent', 'persona': ''},
            follow_redirects=False,
        )
        assert created.status_code == 303
        location = created.headers['location']
        agent_id = location.split('/agents/')[1].split('?')[0]

        chat = client.post(
            f'/agents/{agent_id}/chat',
            data={'message': 'hello from web'},
            follow_redirects=False,
        )
        assert chat.status_code == 303

        page = client.get(f'/agents/{agent_id}')
        assert page.status_code == 200

    assert len(calls) == 1
    assert calls[0]['agent_id'] == agent_id
    assert calls[0]['user_input'] == 'hello from web'
    assert calls[0]['channel'] == 'web'
    assert calls[0]['chat_id'] == 'web-ui'
    assert calls[0]['sender_id'] == 'local-ui'
    assert calls[0]['metadata'].get('source') == 'web-chat'
