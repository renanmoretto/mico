import asyncio
import time
from typing import Any, Callable

from agno.agent import Agent
from agno.models.message import Message
from agno.run.agent import RunOutput

from . import compact
from . import storage
from .config import CONFIG, get_model
from .logging import logger
from .runtime import get_runtime_manager
from .tools import TOOLS


def _tool_hook(function_name: str, function_call: Callable, arguments: dict[str, Any]) -> Any:
    args_preview = {k: repr(v)[:100] for k, v in arguments.items() if k != 'run_context'}
    logger.info(f'[tool] calling {function_name} | args={args_preview}')
    t0 = time.perf_counter()
    try:
        result = function_call(**arguments)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result_preview = repr(result)[:200] if result is not None else 'None'
        logger.info(f'[tool] {function_name} ok | {elapsed_ms:.0f}ms | result={result_preview}')
        return result
    except Exception:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception(f'[tool] {function_name} failed | {elapsed_ms:.0f}ms')
        raise


class Mico:
    def __init__(self):
        self._agent_cache: dict[str, tuple[str, str, Agent]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _build_system_message(self, persona: str) -> str:
        base = (
            'You are Mico, a persistent agent with one continuous brain. '
            'Use memory/message tools when helpful, but only on demand. '
            'You can create, update, and delete semantic memories autonomously. '
            'When searching old chat content, use message search tools.'
        )
        persona_text = persona.strip()
        if not persona_text:
            return base
        return f'{base}\n\nAgent persona:\n{persona_text}'

    def _build_history_input(self, *, agent_id: str, token_budget: int = 20_000) -> list[Message]:
        rows = compact.select_recent_messages_for_context(agent_id=agent_id, token_budget=token_budget)
        history: list[Message] = []
        for row in rows:
            role = str(row.role)
            content = row.content
            if role not in {'user', 'assistant', 'system', 'tool'}:
                continue
            if content is None:
                continue
            text = str(content)
            if not text.strip():
                continue
            history.append(Message(role=role, content=text))
        return history

    async def _get_agent(self, *, agent_row: storage.AgentRecord) -> Agent:
        system_message = self._build_system_message(agent_row.persona)
        model_name = CONFIG.model.openrouter_model
        cached = self._agent_cache.get(agent_row.id)
        if cached and cached[0] == system_message and cached[1] == model_name:
            return cached[2]

        logger.info(f'[agent:{agent_row.id[:8]}] building agent | model={model_name}')
        agent = Agent(
            model=get_model(),
            system_message=system_message,
            tools=TOOLS,
            tool_hooks=[_tool_hook],
        )
        self._agent_cache[agent_row.id] = (system_message, model_name, agent)
        return agent

    def _lock_for_agent(self, agent_id: str) -> asyncio.Lock:
        lock = self._locks.get(agent_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[agent_id] = lock
        return lock

    async def run(
        self,
        *,
        agent_id: str,
        user_input: str | None = None,
        system_input: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        preview = (user_input or system_input or '').strip()[:80]
        logger.info(
            f'[{agent_id[:8]}] message received | channel={channel} chat_id={chat_id} sender={sender_id} | {preview!r}'
        )
        if not (user_input and user_input.strip()) and not (system_input and system_input.strip()):
            raise ValueError('run requires at least one non-empty input (user_input or system_input).')

        lock = self._lock_for_agent(agent_id)
        if lock.locked():
            logger.debug(f'[{agent_id[:8]}] queued — agent is already processing a message')
        async with lock:
            return await self._run_locked(
                agent_id=agent_id,
                user_input=user_input,
                system_input=system_input,
                channel=channel,
                chat_id=chat_id,
                sender_id=sender_id,
                metadata=dict(metadata or {}),
            )

    async def _run_locked(
        self,
        *,
        agent_id: str,
        user_input: str | None,
        system_input: str | None,
        channel: str | None,
        chat_id: str | None,
        sender_id: str | None,
        metadata: dict[str, object],
    ) -> str:
        tag = agent_id[:8]
        t0 = time.perf_counter()

        agent_row = storage.get_agent(agent_id)
        if agent_row is None:
            raise ValueError(f"Agent '{agent_id}' not found.")

        t_compact = time.perf_counter()
        compact.compact_conversation_if_needed(agent_id=agent_id)
        logger.debug(f'[{tag}] compact check done in {(time.perf_counter() - t_compact) * 1000:.0f}ms')

        t_history = time.perf_counter()
        history_input: list[Message] = self._build_history_input(agent_id=agent_id)
        context_msgs = len(history_input)
        logger.debug(
            f'[{tag}] context built: {context_msgs} messages in {(time.perf_counter() - t_history) * 1000:.0f}ms'
        )

        if system_input and system_input.strip():
            history_input.append(Message(role='system', content=system_input.strip()))
        if user_input and user_input.strip():
            history_input.append(Message(role='user', content=user_input.strip()))

        t_agent = time.perf_counter()
        agent = await self._get_agent(agent_row=agent_row)
        cached = agent_row.id in self._agent_cache
        logger.debug(
            f'[{tag}] agent {"from cache" if cached else "rebuilt"} in {(time.perf_counter() - t_agent) * 1000:.0f}ms'
        )

        chunks: list[str] = []
        run_output: RunOutput | None = None
        t_stream_start = time.perf_counter()
        t_first_token: float | None = None

        model_id = CONFIG.model.openrouter_model
        logger.debug(f'[{tag}] starting LLM stream | model={model_id} context_messages={context_msgs + (1 if user_input else 0)}')

        agent_stream = agent.arun(
            input=history_input if history_input else (user_input or system_input or ''),
            session_state={'agent_id': agent_id, 'runtime': get_runtime_manager()},
            stream=True,
            stream_events=True,
            yield_run_output=True,
        )

        async for chunk in agent_stream:
            event = getattr(chunk, 'event', None)
            if event == 'RunContent':
                piece = getattr(chunk, 'content', None)
                if piece:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        logger.debug(
                            f'[{tag}] first token in {(t_first_token - t_stream_start) * 1000:.0f}ms'
                        )
                    chunks.append(str(piece))
                continue

            if isinstance(chunk, RunOutput):
                run_output = chunk

        t_stream_done = time.perf_counter()
        stream_elapsed = t_stream_done - t_stream_start
        logger.debug(
            f'[{tag}] stream done | chunks={len(chunks)} elapsed={stream_elapsed * 1000:.0f}ms'
        )

        content = str(run_output.content) if run_output and run_output.content is not None else ''.join(chunks)

        t_save = time.perf_counter()
        now = int(time.time())
        event_metadata = {
            **metadata,
            **({k: v for k, v in {'channel': channel, 'chat_id': chat_id, 'sender_id': sender_id}.items() if v}),
        }

        def _save(role: str, text: str | None) -> None:
            if text and text.strip():
                storage.add_message(agent_id=agent_id, role=role, content=text.strip(), timestamp=now, metadata=event_metadata)

        _save('system', system_input)
        _save('user', user_input)
        _save('assistant', content if content.strip() else None)
        logger.debug(f'[{tag}] messages saved in {(time.perf_counter() - t_save) * 1000:.0f}ms')

        total_elapsed = time.perf_counter() - t0
        ttft_ms = f'{(t_first_token - t_stream_start) * 1000:.0f}ms' if t_first_token else 'n/a'
        logger.info(
            f'[{tag}] run complete | total={total_elapsed * 1000:.0f}ms'
            f' llm={stream_elapsed * 1000:.0f}ms'
            f' ttft={ttft_ms}'
            f' response_len={len(content)}'
        )
        return content
