from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from . import storage
from .runtime import get_runtime_manager

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = 'config.json'
_DEFAULT_AGENT_CONFIG = {
    'llm': None,
    'channels': {
        'telegram': {
            'enabled': False,
            'bot_token': '',
            'allowed_chat_ids': [],
        }
    }
}
_WEB_CHANNEL_CONFIG = {'default_chat_id': 'web-ui'}


@dataclass(frozen=True)
class AgentChannel:
    agent_id: str
    channel: str
    enabled: bool
    config: dict[str, Any]


class TelegramChannelConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    bot_token: str = ''
    allowed_chat_ids: list[str] = Field(default_factory=list)

    @field_validator('bot_token', mode='before')
    @classmethod
    def _normalize_bot_token(cls, value: Any) -> str:
        return str(value or '').strip()

    @field_validator('allowed_chat_ids', mode='before')
    @classmethod
    def _normalize_allowed_chat_ids(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


class GenericChannelConfigModel(BaseModel):
    model_config = ConfigDict(extra='allow')

    enabled: bool = False


class AgentConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    llm: dict[str, str] | None = None
    channels: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator('llm', mode='before')
    @classmethod
    def _normalize_llm(cls, value: Any) -> dict[str, str] | None:
        if value is None:
            return None
        source = _as_dict(value)
        provider = str(source.get('provider') or 'openrouter').strip().lower() or 'openrouter'
        model = str(source.get('model') or '').strip()
        if provider != 'openrouter':
            raise ValueError("Only the 'openrouter' provider is supported.")
        if not model:
            raise ValueError("'llm.model' must be a non-empty string.")
        return {
            'provider': provider,
            'model': model,
        }

    @field_validator('channels', mode='before')
    @classmethod
    def _channels_must_be_dict(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("'channels' must be an object.")
        return value

    @model_validator(mode='after')
    def _normalize_channels(self) -> 'AgentConfigModel':
        normalized: dict[str, dict[str, Any]] = {
            'telegram': _validate_channel_payload('telegram', self.channels.get('telegram'))
        }
        for raw_name, raw_value in self.channels.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError('Channel names must be non-empty strings.')
            if name == 'telegram':
                continue
            if name == 'web':
                raise ValueError("The 'web' channel is built in and cannot be configured.")
            normalized[name] = _validate_channel_payload(name, raw_value)
        self.channels = normalized
        return self


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return default


def _normalize_string_list(value: Any) -> list[str]:
    items: list[Any]
    if isinstance(value, str):
        items = value.replace(',', '\n').splitlines()
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = []

    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_channel_payload(channel: str, raw: Any) -> dict[str, Any]:
    source = _as_dict(raw)
    normalized = dict(source)
    normalized['enabled'] = _coerce_bool(source.get('enabled', False), False)

    if channel == 'telegram':
        normalized['bot_token'] = str(source.get('bot_token') or '').strip()
        normalized['allowed_chat_ids'] = _normalize_string_list(source.get('allowed_chat_ids'))

    return normalized


def _validate_channel_payload(channel: str, raw: Any) -> dict[str, Any]:
    source = _as_dict(raw)
    if channel == 'telegram':
        return TelegramChannelConfigModel.model_validate(source).model_dump()
    return GenericChannelConfigModel.model_validate(source).model_dump()


def _has_channel_settings(channel: str, *, enabled: bool, config: dict[str, Any]) -> bool:
    if enabled:
        return True
    if channel == 'telegram':
        if str(config.get('bot_token') or '').strip():
            return True
        allowed = config.get('allowed_chat_ids')
        return isinstance(allowed, list) and len(allowed) > 0
    return bool(config)


def _default_payload() -> dict[str, Any]:
    return copy.deepcopy(_DEFAULT_AGENT_CONFIG)


def _normalize_payload(raw: Any) -> dict[str, Any]:
    return AgentConfigModel.model_validate(copy.deepcopy(_as_dict(raw))).model_dump()


def _validation_error_message(exc: ValidationError) -> str:
    issue = exc.errors(include_url=False)[0]
    location = '.'.join(str(part) for part in issue.get('loc', ()) if part != '__root__')
    message = str(issue.get('msg') or 'Invalid config.')
    return f'{location}: {message}' if location else message


def _write_text_atomic(path: Path, text: str) -> None:
    temp_path = path.with_name(f'.{path.name}.tmp')
    temp_path.write_text(text, encoding='utf-8')
    temp_path.replace(path)


async def _config_path(agent_id: str) -> Path:
    workspace = get_runtime_manager().workspace_path(agent_id)
    await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
    return workspace / _CONFIG_FILENAME


async def _write_payload(agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        normalized = _normalize_payload(payload)
    except ValidationError as exc:
        raise ValueError(f'Invalid agent config: {_validation_error_message(exc)}') from exc
    path = await _config_path(agent_id)
    rendered = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=True) + '\n'
    await asyncio.to_thread(_write_text_atomic, path, rendered)
    return copy.deepcopy(normalized)


async def ensure_agent_config(agent_id: str) -> dict[str, Any]:
    path = await _config_path(agent_id)
    if await asyncio.to_thread(path.exists):
        return await get_agent_config(agent_id)
    return await _write_payload(agent_id, _default_payload())


async def get_agent_config(agent_id: str, *, strict: bool = False) -> dict[str, Any]:
    path = await _config_path(agent_id)
    if not await asyncio.to_thread(path.exists):
        return await ensure_agent_config(agent_id)

    try:
        raw = await asyncio.to_thread(path.read_text, encoding='utf-8')
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError(f"Agent config for '{agent_id}' is not valid JSON.") from exc
        logger.warning('Invalid config.json for agent %s: %s', agent_id, exc)
        return _default_payload()

    if not isinstance(parsed, dict):
        if strict:
            raise ValueError(f"Agent config for '{agent_id}' must be a JSON object.")
        logger.warning('Invalid config.json root for agent %s: expected object.', agent_id)
        return _default_payload()

    try:
        return _normalize_payload(parsed)
    except ValidationError as exc:
        if strict:
            raise ValueError(f"Agent config for '{agent_id}' is invalid: {_validation_error_message(exc)}") from exc
        logger.warning('Invalid config.json shape for agent %s: %s', agent_id, _validation_error_message(exc))
        return _default_payload()


async def set_agent_config(agent_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return await _write_payload(agent_id, config)


async def get_agent_llm_config(agent_id: str) -> 'app_config.LLMConfig':
    from . import config as app_config

    payload = await get_agent_config(agent_id)
    raw = _as_dict(payload.get('llm'))
    if not raw:
        return (await app_config.get_app_config()).llm
    return app_config.LLMConfig(
        provider=str(raw.get('provider') or 'openrouter'),
        model=str(raw.get('model') or '').strip(),
    )


async def get_agent_channel(agent_id: str, channel: str) -> AgentChannel | None:
    if channel == 'web':
        return AgentChannel(
            agent_id=agent_id,
            channel='web',
            enabled=True,
            config=dict(_WEB_CHANNEL_CONFIG),
        )

    payload = await get_agent_config(agent_id)
    raw = _as_dict(_as_dict(payload.get('channels')).get(channel))
    if not raw:
        return None

    normalized = _normalize_channel_payload(channel, raw)
    enabled = bool(normalized.pop('enabled', False))
    return AgentChannel(
        agent_id=agent_id,
        channel=channel,
        enabled=enabled,
        config=normalized,
    )


async def list_agent_channels(agent_id: str, *, enabled_only: bool = False) -> list[AgentChannel]:
    payload = await get_agent_config(agent_id)
    rows: list[AgentChannel] = []
    for channel in sorted(_as_dict(payload.get('channels'))):
        row = await get_agent_channel(agent_id, channel)
        if row is None:
            continue
        if enabled_only and not row.enabled:
            continue
        if not enabled_only and not _has_channel_settings(channel, enabled=row.enabled, config=row.config):
            continue
        rows.append(row)
    return rows


async def list_enabled_agent_channels(channel: str | None = None) -> list[AgentChannel]:
    rows = await storage.list_agents()
    matches: list[AgentChannel] = []

    for row in rows:
        if channel is not None:
            current = await get_agent_channel(row.id, channel)
            if current is not None and current.enabled:
                matches.append(current)
            continue

        matches.extend(await list_agent_channels(row.id, enabled_only=True))

    return matches


async def configure_channel(
    *,
    agent_id: str,
    channel: str,
    enabled: bool,
    config: dict[str, object] | None = None,
) -> AgentChannel:
    if channel == 'web':
        raise ValueError("The 'web' channel is built in and cannot be configured.")

    payload = await get_agent_config(agent_id)
    channels = _as_dict(payload.get('channels'))
    current = dict(_as_dict(channels.get(channel)))
    current.update(_as_dict(config))
    current['enabled'] = enabled
    channels[channel] = current
    payload['channels'] = channels
    await set_agent_config(agent_id, payload)
    row = await get_agent_channel(agent_id, channel)
    if row is None:
        raise RuntimeError(f'Failed to configure {channel} for agent {agent_id}.')
    return row
