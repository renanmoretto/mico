from __future__ import annotations

import asyncio

from mico import config, storage


def test_app_config_is_seeded_in_db(tmp_path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'config-seed.db'
        await storage.init_storage(db_path=str(db_path))

        cfg = await config.get_app_config()
        raw = await storage.get_config(key='app')

        assert raw is not None
        assert raw.get('model', {}).get('openrouter_model') == cfg.model.openrouter_model
        assert raw.get('runtime', {}).get('docker_image') == cfg.runtime.docker_image

    asyncio.run(scenario())


def test_update_app_config_persists_changes(tmp_path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / 'config-update.db'
        await storage.init_storage(db_path=str(db_path))

        updated = await config.update_app_config(
            {
                'model': {'openrouter_model': 'openai/gpt-4o-mini'},
                'web': {'telegram_autostart': False},
            }
        )
        reloaded = await config.get_app_config()
        raw = await storage.get_config(key='app')

        assert updated.model.openrouter_model == 'openai/gpt-4o-mini'
        assert reloaded.model.openrouter_model == 'openai/gpt-4o-mini'
        assert reloaded.web.telegram_autostart is False
        assert raw is not None
        assert raw.get('web', {}).get('telegram_autostart') is False

    asyncio.run(scenario())
