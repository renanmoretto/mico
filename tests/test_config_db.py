from __future__ import annotations

from mico import config, storage


def test_app_config_is_seeded_in_db(tmp_path) -> None:
    db_path = tmp_path / 'config-seed.db'
    storage.init_storage(db_path=str(db_path))

    cfg = config.get_app_config()
    raw = storage.get_config(key='app')

    assert raw is not None
    assert raw.get('model', {}).get('openrouter_model') == cfg.model.openrouter_model
    assert raw.get('runtime', {}).get('docker_image') == cfg.runtime.docker_image


def test_update_app_config_persists_changes(tmp_path) -> None:
    db_path = tmp_path / 'config-update.db'
    storage.init_storage(db_path=str(db_path))

    updated = config.update_app_config(
        {
            'model': {'openrouter_model': 'openai/gpt-4o-mini'},
            'web': {'telegram_autostart': False},
        }
    )
    reloaded = config.get_app_config()
    raw = storage.get_config(key='app')

    assert updated.model.openrouter_model == 'openai/gpt-4o-mini'
    assert reloaded.model.openrouter_model == 'openai/gpt-4o-mini'
    assert reloaded.web.telegram_autostart is False
    assert raw is not None
    assert raw.get('web', {}).get('telegram_autostart') is False
