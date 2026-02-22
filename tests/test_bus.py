from __future__ import annotations

import asyncio

import pytest

from mico.bus import AgentMessageWorker, MessageBus, OutboundMessageWorker
from mico.channels import ChannelManager
from mico.messages import InboundMessage, OutboundMessage, send, unregister_sender


class _FakeMico:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(
        self,
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
        self.calls.append(
            {
                'agent_id': agent_id,
                'user_input': user_input,
                'system_input': system_input,
                'quiet': quiet,
                'channel': channel,
                'chat_id': chat_id,
                'sender_id': sender_id,
                'metadata': dict(metadata or {}),
            }
        )
        await asyncio.sleep(0)
        return f'echo:{user_input}'


def test_message_bus_roundtrip_dispatches_sender() -> None:
    async def scenario() -> None:
        fake = _FakeMico()
        bus = MessageBus()
        inbound_worker = AgentMessageWorker(bus=bus, mico=fake)
        outbound_worker = OutboundMessageWorker(bus=bus)
        sent: list[OutboundMessage] = []

        async def capture_sender(message: OutboundMessage) -> None:
            sent.append(message)

        from mico.messages import register_sender

        await register_sender('test', capture_sender)
        await inbound_worker.start()
        await outbound_worker.start()
        try:
            response = await bus.publish_inbound(
                InboundMessage(
                    agent_id='agent-1',
                    channel='test',
                    sender_id='user-1',
                    chat_id='chat-1',
                    content='hello',
                    metadata={'source': 'pytest'},
                ),
                wait_response=True,
                timeout_seconds=2.0,
            )
            assert response == 'echo:hello'
            assert len(fake.calls) == 1
            assert fake.calls[0]['channel'] == 'test'
            assert fake.calls[0]['chat_id'] == 'chat-1'
            assert len(sent) == 1
            assert sent[0].content == 'echo:hello'
            assert sent[0].agent_id == 'agent-1'
            assert sent[0].channel == 'test'
        finally:
            await outbound_worker.stop()
            await inbound_worker.stop()
            await unregister_sender('test')

    asyncio.run(scenario())


def test_channel_manager_registers_and_unreg_web_sender() -> None:
    async def scenario() -> None:
        manager = ChannelManager(bus=MessageBus(), telegram_enabled=False)
        message = OutboundMessage(
            agent_id='agent-1',
            channel='web',
            chat_id='web-ui',
            content='hi',
        )

        await manager.start()
        try:
            await send('web', message)
        finally:
            await manager.stop()

        with pytest.raises(ValueError, match='No sender registered'):
            await send('web', message)

    asyncio.run(scenario())
