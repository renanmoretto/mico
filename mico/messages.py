from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .logging import logger


ChannelName = str
SenderCallable = Callable[['OutboundMessage'], Awaitable[None]]


@dataclass(slots=True)
class InboundMessage:
    agent_id: str
    channel: ChannelName
    sender_id: str
    chat_id: str
    content: str
    message_id: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        # Addressing key for transport-level routing; memory remains agent-wide.
        return f'{self.channel}:{self.chat_id}'


@dataclass(slots=True)
class OutboundMessage:
    agent_id: str
    channel: ChannelName
    chat_id: str
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reply_to_message_id: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = field(default_factory=dict)


_sender_registry: dict[str, SenderCallable] = {}
_registry_lock = asyncio.Lock()


async def register_sender(channel: ChannelName, sender: SenderCallable) -> None:
    async with _registry_lock:
        _sender_registry[channel] = sender
    logger.info(f'Registered sender for channel: {channel}')


async def unregister_sender(channel: ChannelName) -> None:
    async with _registry_lock:
        _sender_registry.pop(channel, None)
    logger.info(f'Unregistered sender for channel: {channel}')


async def send(channel: ChannelName, message: OutboundMessage) -> None:
    logger.debug(f'Sending message to channel={channel}, chat_id={message.chat_id}')
    async with _registry_lock:
        sender = _sender_registry.get(channel)

    if sender is None:
        logger.error(f'No sender registered for channel: {channel}')
        raise ValueError(f'No sender registered for channel {channel}.')
    if message.channel != channel:
        logger.error(f'Channel mismatch: expected {channel}, got {message.channel}')
        raise ValueError(f'OutboundMessage.channel mismatch: expected {channel}, got {message.channel}.')
    await sender(message)
    logger.info(f'Message sent to channel={channel}, chat_id={message.chat_id}')
