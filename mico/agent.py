import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable

from agno.agent import Agent
from agno.models.message import Message
from agno.run.agent import RunOutput

from . import agent_config
from . import compact
from . import storage
from .config import get_model
from .logging import logger
from .messages import OutboundMessage
from .prompts import SYSTEM_PROMPT
from .runtime import get_runtime_manager
from .tools import TOOLS

# ── Event broadcasting (stream events to UI) ──

_event_subscribers: dict[str, set[asyncio.Queue]] = {}


def subscribe_events(agent_id: str) -> asyncio.Queue:
    if agent_id not in _event_subscribers:
        _event_subscribers[agent_id] = set()
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _event_subscribers[agent_id].add(q)
    return q


def unsubscribe_events(agent_id: str, q: asyncio.Queue) -> None:
    subs = _event_subscribers.get(agent_id)
    if subs:
        subs.discard(q)
        if not subs:
            del _event_subscribers[agent_id]


def _publish_event(agent_id: str, event: dict[str, object]) -> None:
    subs = _event_subscribers.get(agent_id)
    if not subs:
        return
    for q in subs:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


async def _tool_hook(function_name: str, function_call: Callable, arguments: dict[str, Any]) -> Any:
    args_preview = {k: repr(v)[:100] for k, v in arguments.items() if k != 'run_context'}
    logger.info(f'[tool] calling {function_name} | args={args_preview}')
    t0 = time.perf_counter()
    try:
        result = await function_call(**arguments)
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
        self._locks: dict[str, asyncio.Lock] = {}
        self.bus: Any = None  # set by web.py after bus creation

    async def _build_system_message(self, agent_id: str) -> str:
        prompt = SYSTEM_PROMPT.format(date=datetime.now().strftime('%A, %B %d, %Y'))
        runtime = get_runtime_manager()
        roots = await runtime.list_root_names(agent_id)
        prompt += '\n\n## Available File Roots\n\n'
        prompt += 'Use the `root` argument with workspace file tools.\n'
        prompt += 'Current roots for this agent: ' + ', '.join(roots) + '.'
        for filename in ('SOUL.md', 'MEMORY.md'):
            try:
                content = await runtime.read_file(agent_id=agent_id, path=filename)
                if content and content.strip():
                    prompt += f'\n\n---\n\n# {filename}\n\n{content.strip()}'
            except Exception:
                pass
        return prompt

    async def _build_history_input(self, *, agent_id: str, token_budget: int = 20_000) -> list[Message]:
        rows = await compact.select_recent_messages_for_context(agent_id=agent_id, token_budget=token_budget)
        history: list[Message] = []
        for row in rows:
            role = str(row.role)
            if role not in {'user', 'assistant', 'system', 'tool'}:
                continue
            has_content = bool(row.content and row.content.strip())
            has_tool_calls = bool(row.tool_calls)
            if not has_content and not has_tool_calls:
                continue
            kwargs: dict[str, Any] = {'role': role}
            if has_content:
                kwargs['content'] = row.content.strip()
            if row.tool_call_id:
                kwargs['tool_call_id'] = row.tool_call_id
            if row.tool_calls:
                kwargs['tool_calls'] = row.tool_calls
            tool_name = str(row.metadata.get('tool_name') or '').strip()
            if role == 'tool' and tool_name:
                kwargs['tool_name'] = tool_name
            if row.metadata.get('tool_call_error'):
                kwargs['tool_call_error'] = True
            history.append(Message(**kwargs))
        return history

    async def _get_agent(self, *, agent_id: str) -> Agent:
        system_message = await self._build_system_message(agent_id)
        llm = await agent_config.get_agent_llm_config(agent_id)
        model_name = f'{llm.provider}:{llm.model}'
        logger.info(f'[agent:{agent_id[:8]}] building agent | model={model_name}')
        return Agent(
            model=await get_model(llm),
            system_message=system_message,
            tools=TOOLS,
            tool_hooks=[_tool_hook],
        )

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

        agent_row = await storage.get_agent(agent_id)
        if agent_row is None:
            raise ValueError(f"Agent '{agent_id}' not found.")

        _publish_event(agent_id, {'type': 'run_start'})

        t_compact = time.perf_counter()
        await compact.compact_conversation_if_needed(agent_id=agent_id)
        logger.debug(f'[{tag}] compact check done in {(time.perf_counter() - t_compact) * 1000:.0f}ms')

        t_history = time.perf_counter()
        history_input: list[Message] = await self._build_history_input(agent_id=agent_id)
        context_msgs = len(history_input)
        logger.debug(
            f'[{tag}] context built: {context_msgs} messages in {(time.perf_counter() - t_history) * 1000:.0f}ms'
        )

        if system_input and system_input.strip():
            history_input.append(Message(role='system', content=system_input.strip()))
        if user_input and user_input.strip():
            history_input.append(Message(role='user', content=user_input.strip()))

        t_agent = time.perf_counter()
        agent = await self._get_agent(agent_id=agent_id)
        logger.debug(f'[{tag}] agent built in {(time.perf_counter() - t_agent) * 1000:.0f}ms')

        pending_outbound: list[OutboundMessage] = []

        chunks: list[str] = []
        run_output: RunOutput | None = None
        t_stream_start = time.perf_counter()
        t_first_token: float | None = None

        llm = await agent_config.get_agent_llm_config(agent_id)
        model_id = f'{llm.provider}:{llm.model}'
        logger.debug(f'[{tag}] starting LLM stream | model={model_id} context_messages={context_msgs + (1 if user_input else 0)}')

        agent_stream = agent.arun(
            input=history_input if history_input else (user_input or system_input or ''),
            session_state={
                'agent_id': agent_id,
                'runtime': get_runtime_manager(),
                'pending_outbound': pending_outbound,
            },
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

            if event in ('ToolCallStarted', 'ToolCallCompleted', 'ToolCallError'):
                tool = getattr(chunk, 'tool', None)
                if tool:
                    tool_args = {k: repr(v)[:100] for k, v in (tool.tool_args or {}).items()}
                    if event == 'ToolCallStarted':
                        _publish_event(agent_id, {
                            'type': 'tool_start',
                            'call_id': tool.tool_call_id or '',
                            'name': tool.tool_name or '',
                            'args': tool_args,
                        })
                    elif event == 'ToolCallCompleted':
                        elapsed = round(tool.metrics.duration * 1000) if tool.metrics and tool.metrics.duration else 0
                        _publish_event(agent_id, {
                            'type': 'tool_end',
                            'call_id': tool.tool_call_id or '',
                            'name': tool.tool_name or '',
                            'elapsed_ms': elapsed,
                            'ok': True,
                        })
                    else:  # ToolCallError
                        elapsed = round(tool.metrics.duration * 1000) if tool.metrics and tool.metrics.duration else 0
                        _publish_event(agent_id, {
                            'type': 'tool_end',
                            'call_id': tool.tool_call_id or '',
                            'name': tool.tool_name or '',
                            'elapsed_ms': elapsed,
                            'ok': False,
                        })
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
        ts_now = int(time.time())
        event_metadata = {
            **metadata,
            **({k: v for k, v in {'channel': channel, 'chat_id': chat_id, 'sender_id': sender_id}.items() if v}),
        }

        async def _save(
            role: str,
            *,
            content: str | None = None,
            tool_call_id: str | None = None,
            tool_calls: list[dict[str, Any]] | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            clean_content = content.strip() if content and content.strip() else None
            if clean_content is None and not tool_calls:
                return
            message_metadata = {**event_metadata, **(metadata or {})}
            await storage.add_message(
                agent_id=agent_id,
                role=role,
                content=clean_content,
                timestamp=ts_now,
                tool_call_id=tool_call_id.strip() if tool_call_id and tool_call_id.strip() else None,
                tool_calls=tool_calls or None,
                metadata=message_metadata,
            )

        await _save('system', content=system_input)
        await _save('user', content=user_input)

        # Save tool calls and results from the run
        if run_output and run_output.messages:
            for msg in run_output.messages:
                if msg.role == 'assistant' and msg.tool_calls:
                    assistant_content = msg.get_content_string() if msg.content else None
                    await _save(
                        'assistant',
                        content=assistant_content,
                        tool_calls=json.loads(json.dumps(msg.tool_calls, default=str)),
                    )
                elif msg.role == 'tool':
                    tool_content = msg.get_content_string() if msg.content else None
                    tool_meta: dict[str, Any] = {}
                    if msg.tool_name:
                        tool_meta['tool_name'] = msg.tool_name
                    if msg.tool_call_error:
                        tool_meta['tool_call_error'] = True
                    await _save(
                        'tool',
                        content=tool_content,
                        tool_call_id=msg.tool_call_id,
                        metadata=tool_meta,
                    )

        await _save('assistant', content=content if content.strip() else None)
        logger.debug(f'[{tag}] messages saved in {(time.perf_counter() - t_save) * 1000:.0f}ms')

        if self.bus and pending_outbound:
            for msg in pending_outbound:
                await self.bus.publish_outbound(msg)
            logger.debug(f'[{tag}] published {len(pending_outbound)} outbound message(s) from tools')

        total_elapsed = time.perf_counter() - t0
        ttft_ms = f'{(t_first_token - t_stream_start) * 1000:.0f}ms' if t_first_token else 'n/a'
        logger.info(
            f'[{tag}] run complete | total={total_elapsed * 1000:.0f}ms'
            f' llm={stream_elapsed * 1000:.0f}ms'
            f' ttft={ttft_ms}'
            f' response_len={len(content)}'
        )
        _publish_event(agent_id, {'type': 'run_end'})
        return content
