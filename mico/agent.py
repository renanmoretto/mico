import time

import typer
from agno.agent import Agent
from agno.models.message import Message
from agno.run.agent import RunOutput

from . import compact
from .config import MODEL
from . import storage
from .tools import TOOLS


class Mico:
    def __init__(self):
        self.agent: Agent | None = None

    def _ui_status(self, message: str) -> None:
        typer.echo(typer.style(f'  {message}', fg=typer.colors.BRIGHT_BLACK))

    def _ui_tool(self, prefix: str, name: str, args: str = '') -> None:
        typer.echo(typer.style(f'  [{prefix}] {name}{args}', fg=typer.colors.YELLOW))

    def _ui_agent_prefix(self) -> str:
        return typer.style('agent', fg=typer.colors.CYAN, bold=True) + ': '

    def _build_system_message(self) -> str:
        return (
            'You are Mico, a persistent agent with one continuous brain. '
            'Use memory/message tools when helpful, but only on demand. '
            'You can create, update, and delete semantic memories autonomously. '
            'When searching old chat content, use message search tools. '
            'When the user asks for future reminders/tasks, create a cron with the cron tools.'
        )

    def _build_history_input(self, token_budget: int = 20_000) -> list[Message]:
        rows = compact.select_recent_messages_for_context(token_budget=token_budget)
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

    async def _get_agent(self) -> Agent:
        if self.agent is None:
            self.agent = Agent(
                model=MODEL,
                system_message=self._build_system_message(),
                tools=TOOLS,
            )
        return self.agent

    async def run(
        self,
        user_input: str | None = None,
        system_input: str | None = None,
        quiet: bool = False,
    ) -> str:
        if not (user_input and user_input.strip()) and not (system_input and system_input.strip()):
            raise ValueError('run requires at least one non-empty input (user_input or system_input).')

        compaction_result = compact.compact_conversation_if_needed()
        if compaction_result.triggered:
            if not quiet:
                self._ui_status(
                    f'context compacted: -{compaction_result.compacted_tokens} tokens, '
                    f'memory={compaction_result.memory_name}'
                )
        history_input: list[Message] = self._build_history_input()
        if system_input and system_input.strip():
            history_input.append(Message(role='system', content=system_input.strip()))
        if user_input and user_input.strip():
            history_input.append(Message(role='user', content=user_input.strip()))

        agent = await self._get_agent()
        agent_stream = agent.arun(
            input=history_input if history_input else (user_input or system_input or ''),
            stream=True,
            stream_events=True,
            yield_run_output=True,
        )

        chunks: list[str] = []
        run_output: RunOutput | None = None
        printed_content = False
        started_streaming_content = False
        started_at = time.perf_counter()

        async for chunk in agent_stream:
            event = getattr(chunk, 'event', None)
            if event in ('RunStarted', 'ModelRequestStarted'):
                continue

            if event == 'RunContent':
                piece = getattr(chunk, 'content', None)
                if piece:
                    text = str(piece)
                    chunks.append(text)
                    printed_content = True
                    if not quiet and not started_streaming_content:
                        typer.echo(self._ui_agent_prefix(), nl=False)
                        started_streaming_content = True
                    if not quiet:
                        typer.echo(text, nl=False)
                continue

            if event in {'ToolCallStarted', 'ToolCallCompleted', 'ToolCallError'}:
                tool = getattr(chunk, 'tool', None)
                tool_name = getattr(tool, 'tool_name', 'tool')
                tool_args = getattr(tool, 'tool_args', None)
                args = f' {tool_args}' if tool_args else ''
                if not quiet:
                    if event == 'ToolCallStarted':
                        self._ui_tool('tool', tool_name, args)
                    elif event == 'ToolCallCompleted':
                        self._ui_tool('tool ok', tool_name)
                    else:
                        self._ui_tool('tool err', tool_name)
                continue

            if isinstance(chunk, RunOutput):
                run_output = chunk

        content = str(run_output.content) if run_output and run_output.content is not None else ''.join(chunks)
        if not quiet and not printed_content and content:
            typer.echo(self._ui_agent_prefix() + content, nl=False)
        elapsed = time.perf_counter() - started_at
        if not quiet:
            typer.echo()
            self._ui_status(f'done in {elapsed:.1f}s')

        now = int(time.time())
        if system_input and system_input.strip():
            storage.add_message(role='system', content=system_input.strip(), timestamp=now)
        if user_input and user_input.strip():
            storage.add_message(role='user', content=user_input.strip(), timestamp=now)

        if content.strip():
            storage.add_message(role='assistant', content=content, timestamp=now)
        return content
