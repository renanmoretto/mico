from __future__ import annotations

import copy
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv

from . import storage

load_dotenv()


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {'1', 'true', 'yes', 'on'}:
            return True
        if v in {'0', 'false', 'no', 'off'}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    return default


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    return default


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ModelDefaults:
    openrouter_model: str = _env_str('MICO_MODEL', 'minimax/minimax-m2.5')


@dataclass(frozen=True)
class TelegramDefaults:
    enabled: bool = _env_bool('MICO_TELEGRAM_ENABLED', False)
    poll_timeout_seconds: int = _env_int('MICO_TELEGRAM_POLL_TIMEOUT_SECONDS', 30)
    poll_interval_seconds: float = _env_float('MICO_TELEGRAM_POLL_INTERVAL_SECONDS', 1.0)
    drop_pending_updates: bool = _env_bool('MICO_TELEGRAM_DROP_PENDING_UPDATES', True)


def _default_agents_dir() -> str:
    from .paths import agents_dir

    return agents_dir()


@dataclass(frozen=True)
class RuntimeDefaults:
    base_dir: str = _env_str('MICO_RUNTIME_BASE_DIR', _default_agents_dir())
    docker_enabled: bool = _env_bool('MICO_DOCKER_ENABLED', True)
    docker_image: str = _env_str('MICO_DOCKER_IMAGE', 'mico-agent:latest')
    idle_stop_seconds: int = _env_int('MICO_DOCKER_IDLE_STOP_SECONDS', 900)


@dataclass(frozen=True)
class WebDefaults:
    telegram_autostart: bool = _env_bool('MICO_WEB_TELEGRAM_AUTOSTART', True)


@dataclass(frozen=True)
class AppConfig:
    model: ModelDefaults = field(default_factory=ModelDefaults)
    telegram: TelegramDefaults = field(default_factory=TelegramDefaults)
    runtime: RuntimeDefaults = field(default_factory=RuntimeDefaults)
    web: WebDefaults = field(default_factory=WebDefaults)


_APP_CONFIG_KEY = 'app'
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: tuple[str, OpenRouter] | None = None
_DEFAULT_CONFIG = AppConfig()


def _default_payload() -> dict[str, Any]:
    cfg = _DEFAULT_CONFIG
    return {
        'model': {
            'openrouter_model': cfg.model.openrouter_model,
        },
        'telegram': {
            'enabled': cfg.telegram.enabled,
            'poll_timeout_seconds': cfg.telegram.poll_timeout_seconds,
            'poll_interval_seconds': cfg.telegram.poll_interval_seconds,
            'drop_pending_updates': cfg.telegram.drop_pending_updates,
        },
        'runtime': {
            'base_dir': cfg.runtime.base_dir,
            'docker_enabled': cfg.runtime.docker_enabled,
            'docker_image': cfg.runtime.docker_image,
            'idle_stop_seconds': cfg.runtime.idle_stop_seconds,
        },
        'web': {
            'telegram_autostart': cfg.web.telegram_autostart,
        },
    }


def _load_raw_payload() -> dict[str, Any]:
    defaults = _default_payload()
    if not storage.is_initialized():
        return copy.deepcopy(defaults)

    current = storage.get_config(key=_APP_CONFIG_KEY)
    if current is None:
        storage.upsert_config(key=_APP_CONFIG_KEY, config=defaults)
        return copy.deepcopy(defaults)

    return _deep_merge(defaults, _as_dict(current))


def _parse_app_config(raw: dict[str, Any]) -> AppConfig:
    D = _DEFAULT_CONFIG
    m = _as_dict(raw.get('model'))
    t = _as_dict(raw.get('telegram'))
    r = _as_dict(raw.get('runtime'))
    w = _as_dict(raw.get('web'))
    return AppConfig(
        model=ModelDefaults(
            openrouter_model=str(m.get('openrouter_model') or D.model.openrouter_model).strip()
            or D.model.openrouter_model,
        ),
        telegram=TelegramDefaults(
            enabled=_coerce_bool(t.get('enabled', D.telegram.enabled), D.telegram.enabled),
            poll_timeout_seconds=max(5, _coerce_int(t.get('poll_timeout_seconds'), D.telegram.poll_timeout_seconds)),
            poll_interval_seconds=max(
                0.05, _coerce_float(t.get('poll_interval_seconds'), D.telegram.poll_interval_seconds)
            ),
            drop_pending_updates=_coerce_bool(
                t.get('drop_pending_updates', D.telegram.drop_pending_updates), D.telegram.drop_pending_updates
            ),
        ),
        runtime=RuntimeDefaults(
            base_dir=str(r.get('base_dir') or D.runtime.base_dir).strip() or D.runtime.base_dir,
            docker_enabled=_coerce_bool(r.get('docker_enabled', D.runtime.docker_enabled), D.runtime.docker_enabled),
            docker_image=str(r.get('docker_image') or D.runtime.docker_image).strip() or D.runtime.docker_image,
            idle_stop_seconds=max(30, _coerce_int(r.get('idle_stop_seconds'), D.runtime.idle_stop_seconds)),
        ),
        web=WebDefaults(
            telegram_autostart=_coerce_bool(
                w.get('telegram_autostart', D.web.telegram_autostart), D.web.telegram_autostart
            ),
        ),
    )


def get_app_config() -> AppConfig:
    return _parse_app_config(_load_raw_payload())


def get_app_config_payload() -> dict[str, Any]:
    return copy.deepcopy(_load_raw_payload())


def set_app_config(config: dict[str, Any]) -> AppConfig:
    if not storage.is_initialized():
        raise RuntimeError('Storage is not initialized. Call init_storage() before setting app config.')
    payload = _deep_merge(_default_payload(), _as_dict(config))
    storage.upsert_config(key=_APP_CONFIG_KEY, config=payload)
    with _MODEL_CACHE_LOCK:
        global _MODEL_CACHE
        _MODEL_CACHE = None
    return _parse_app_config(payload)


def update_app_config(patch: dict[str, Any]) -> AppConfig:
    current = _load_raw_payload()
    next_payload = _deep_merge(current, _as_dict(patch))
    return set_app_config(next_payload)


def get_model() -> OpenRouter:
    model_name = get_app_config().model.openrouter_model
    with _MODEL_CACHE_LOCK:
        global _MODEL_CACHE
        if _MODEL_CACHE is not None and _MODEL_CACHE[0] == model_name:
            return _MODEL_CACHE[1]
        model = OpenRouter(model_name)
        _MODEL_CACHE = (model_name, model)
        return model


class _ConfigProxy:
    @property
    def model(self) -> ModelDefaults:
        return get_app_config().model

    @property
    def telegram(self) -> TelegramDefaults:
        return get_app_config().telegram

    @property
    def runtime(self) -> RuntimeDefaults:
        return get_app_config().runtime

    @property
    def web(self) -> WebDefaults:
        return get_app_config().web


CONFIG = _ConfigProxy()
