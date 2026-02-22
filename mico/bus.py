from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from .agent import Mico
from .logging import logger
from .messages import InboundMessage, OutboundMessage, send


@dataclass(slots=True)
class InboundEnvelope:
    id: str
    message: InboundMessage
    enqueued_at: int
    response_future: asyncio.Future[str] | None = None


class MessageBus:
    def __init__(self, *, inbound_maxsize: int = 1_000, outbound_maxsize: int = 1_000):
        self._inbound: asyncio.Queue[InboundEnvelope] = asyncio.Queue(maxsize=inbound_maxsize)
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=outbound_maxsize)

    async def publish_inbound(
        self,
        message: InboundMessage,
        *,
        wait_response: bool = False,
        timeout_seconds: float | None = None,
    ) -> str | None:
        response_future: asyncio.Future[str] | None = None
        if wait_response:
            response_future = asyncio.get_running_loop().create_future()

        envelope = InboundEnvelope(
            id=str(uuid.uuid4()),
            message=message,
            enqueued_at=int(time.time()),
            response_future=response_future,
        )
        await self._inbound.put(envelope)

        if response_future is None:
            return None
        if timeout_seconds is None:
            return await response_future
        return await asyncio.wait_for(response_future, timeout=timeout_seconds)

    async def publish_outbound(self, message: OutboundMessage) -> None:
        await self._outbound.put(message)

    async def next_inbound(self) -> InboundEnvelope:
        return await self._inbound.get()

    async def next_outbound(self) -> OutboundMessage:
        return await self._outbound.get()


class AgentMessageWorker:
    def __init__(self, *, bus: MessageBus, mico: Mico, max_parallel: int = 16):
        self._bus = bus
        self._mico = mico
        self._max_parallel = max(1, int(max_parallel))
        self._running = False
        self._consumer_task: asyncio.Task | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(self._max_parallel)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop(), name='bus-inbound-consumer')

    async def stop(self) -> None:
        self._running = False

        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        if not self._active_tasks:
            return

        tasks = list(self._active_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _consume_loop(self) -> None:
        while self._running:
            envelope = await self._bus.next_inbound()
            await self._semaphore.acquire()
            task = asyncio.create_task(
                self._process_envelope(envelope),
                name=f'bus-inbound-{envelope.message.agent_id}',
            )
            self._active_tasks.add(task)
            task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._active_tasks.discard(task)
        self._semaphore.release()
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception(f'Inbound worker task failed: {exc}')

    async def _process_envelope(self, envelope: InboundEnvelope) -> None:
        message = envelope.message
        try:
            response = await self._mico.run(
                agent_id=message.agent_id,
                user_input=message.content,
                channel=message.channel,
                chat_id=message.chat_id,
                sender_id=message.sender_id,
                metadata=message.metadata,
            )
        except Exception as exc:
            logger.exception(f'Inbound processing failed for agent {message.agent_id}: {exc}')
            response = f'Error while processing your message: {exc}'

        if response.strip():
            outbound = OutboundMessage(
                agent_id=message.agent_id,
                channel=message.channel,
                chat_id=message.chat_id,
                content=response,
            )
            await self._bus.publish_outbound(outbound)

        if envelope.response_future is not None and not envelope.response_future.done():
            envelope.response_future.set_result(response)


class OutboundMessageWorker:
    def __init__(self, *, bus: MessageBus):
        self._bus = bus
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='bus-outbound-consumer')

    async def stop(self) -> None:
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while self._running:
            message = await self._bus.next_outbound()
            try:
                await send(message.channel, message)
            except Exception as exc:
                logger.exception(f'Outbound send failed for channel={message.channel} agent={message.agent_id}: {exc}')
