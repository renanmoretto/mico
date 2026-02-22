from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from . import storage
from .bus import MessageBus
from .config import CONFIG
from .messages import InboundMessage, OutboundMessage, register_sender, unregister_sender

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from telegram import BotCommand, Update
    from telegram.ext import Application


class ChannelAdapter(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send(self, message: OutboundMessage) -> None: ...


@dataclass(frozen=True)
class TelegramAgentConfig:
    bot_token: str
    allowed_chat_ids: set[str]
    poll_timeout_seconds: int
    poll_interval_seconds: float
    proxy: str | None


class TelegramChannelAdapter:
    def __init__(
        self,
        *,
        agent_id: str,
        config: TelegramAgentConfig,
        bus: MessageBus,
    ):
        self.agent_id = agent_id
        self.config = config
        self._bus = bus
        self._running = False
        self._app: Application | None = None

    async def start(self) -> None:
        if self._running:
            return
        if not self.config.bot_token.strip():
            raise ValueError(f'Agent {self.agent_id} missing Telegram bot token.')

        self._running = True
        try:
            from telegram import BotCommand
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
            from telegram.request import HTTPXRequest

            req = HTTPXRequest(
                connection_pool_size=16,
                pool_timeout=5.0,
                connect_timeout=30.0,
                read_timeout=float(max(5, self.config.poll_timeout_seconds)),
            )
            builder = (
                Application.builder()
                .token(self.config.bot_token)
                .request(req)
                .get_updates_request(req)
            )
            if self.config.proxy:
                builder = builder.proxy(self.config.proxy).get_updates_proxy(self.config.proxy)

            self._app = builder.build()
            self._app.add_error_handler(self._on_error)
            self._app.add_handler(CommandHandler('start', self._on_start))
            self._app.add_handler(CommandHandler('new', self._forward_command))
            self._app.add_handler(CommandHandler('help', self._forward_command))
            self._app.add_handler(
                MessageHandler(
                    (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.Document.ALL)
                    & ~filters.COMMAND,
                    self._on_message,
                )
            )

            await self._app.initialize()
            await self._app.start()

            try:
                await self._app.bot.set_my_commands(
                    [
                        BotCommand('start', 'Start the bot'),
                        BotCommand('new', 'Start a new conversation'),
                        BotCommand('help', 'Show available commands'),
                    ]
                )
            except Exception as exc:
                logger.warning(
                    'Telegram set_my_commands failed for agent %s: %s',
                    self.agent_id,
                    exc,
                )

            if self._app.updater is None:
                raise RuntimeError('Telegram updater is not available.')
            await self._app.updater.start_polling(
                allowed_updates=['message'],
                drop_pending_updates=CONFIG.telegram.drop_pending_updates,
                timeout=self.config.poll_timeout_seconds,
                poll_interval=self.config.poll_interval_seconds,
            )
            logger.info('Telegram adapter started for agent %s.', self.agent_id)
        except ImportError as exc:
            self._running = False
            raise RuntimeError(
                'python-telegram-bot is not installed. Install python-telegram-bot[socks].'
            ) from exc
        except Exception:
            self._running = False
            await self._cleanup_app()
            raise

    async def stop(self) -> None:
        self._running = False
        await self._cleanup_app()

    async def send(self, message: OutboundMessage) -> None:
        if message.agent_id != self.agent_id:
            raise ValueError(
                f'Adapter agent mismatch (expected {self.agent_id}, got {message.agent_id}).'
            )
        if self._app is None:
            raise RuntimeError(f'Telegram adapter for agent {self.agent_id} is not running.')

        for chunk in _split_message(message.content):
            payload: dict[str, Any] = {'chat_id': message.chat_id}
            if message.reply_to_message_id:
                try:
                    payload['reply_to_message_id'] = int(message.reply_to_message_id)
                except ValueError:
                    payload['reply_to_message_id'] = message.reply_to_message_id
            try:
                await self._app.bot.send_message(
                    **payload,
                    text=_markdown_to_telegram_html(chunk),
                    parse_mode='HTML',
                )
            except Exception as exc:
                logger.warning(
                    'Telegram HTML parse failed for agent %s, falling back to plain text: %s',
                    self.agent_id,
                    exc,
                )
                await self._app.bot.send_message(**payload, text=chunk)

    async def _cleanup_app(self) -> None:
        app = self._app
        self._app = None
        if app is None:
            return
        try:
            if app.updater is not None:
                await app.updater.stop()
        except Exception:
            pass
        try:
            await app.stop()
        except Exception:
            pass
        try:
            await app.shutdown()
        except Exception:
            pass

    async def _on_start(self, update: Update, context: Any) -> None:
        if update.message is None or update.effective_user is None:
            return
        await update.message.reply_text(
            f"Hi {update.effective_user.first_name or 'there'}! "
            'This bot is connected to Mico.'
        )

    async def _forward_command(self, update: Update, context: Any) -> None:
        if update.message is None:
            return
        text = str(update.message.text or '').strip()
        if not text:
            return
        await self._publish_update(update=update, content=text)

    async def _on_message(self, update: Update, context: Any) -> None:
        if update.message is None:
            return
        message = update.message
        content = str(message.text or message.caption or '').strip()
        if not content:
            if message.voice is not None:
                content = '[voice message]'
            elif message.audio is not None:
                content = '[audio message]'
            elif message.photo:
                content = '[photo]'
            elif message.document is not None:
                filename = str(message.document.file_name or '').strip()
                content = f"[document:{filename}]" if filename else '[document]'
        if not content:
            return
        await self._publish_update(update=update, content=content)

    async def _publish_update(self, *, update: Update, content: str) -> None:
        message = update.message
        sender = update.effective_user
        chat = update.effective_chat
        if message is None or sender is None or chat is None:
            return

        chat_id = str(chat.id).strip()
        sender_id = str(sender.id).strip()
        if not chat_id or not sender_id:
            return
        if self.config.allowed_chat_ids and chat_id not in self.config.allowed_chat_ids:
            return

        inbound = InboundMessage(
            agent_id=self.agent_id,
            channel='telegram',
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            message_id=str(message.message_id),
            metadata={
                'update_id': update.update_id,
                'message_id': message.message_id,
                'username': sender.username,
                'first_name': sender.first_name,
                'last_name': sender.last_name,
                'chat_type': chat.type,
            },
        )
        await self._bus.publish_inbound(inbound)

    async def _on_error(self, update: object, context: Any) -> None:
        logger.exception('Telegram handler error for agent %s: %s', self.agent_id, context.error)


class TelegramChannelService:
    def __init__(self, *, bus: MessageBus):
        self._bus = bus
        self._adapters: dict[str, TelegramChannelAdapter] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        rows = storage.list_enabled_agent_channels(channel='telegram')
        for row in rows:
            cfg = _telegram_config_from_channel(row.config)
            if not cfg.bot_token:
                logger.warning('Skipping telegram channel for agent %s: bot_token missing.', row.agent_id)
                continue

            adapter = TelegramChannelAdapter(
                agent_id=row.agent_id,
                config=cfg,
                bus=self._bus,
            )
            self._adapters[row.agent_id] = adapter

        await register_sender('telegram', self.send)

        for adapter in self._adapters.values():
            await adapter.start()

        logger.info('Telegram service started with %s adapter(s).', len(self._adapters))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        await unregister_sender('telegram')
        adapters = list(self._adapters.values())
        self._adapters.clear()

        for adapter in adapters:
            await adapter.stop()

        logger.info('Telegram service stopped.')

    async def reload(self) -> None:
        if not self._running:
            return
        await self.stop()
        await self.start()

    async def send(self, message: OutboundMessage) -> None:
        adapter = self._adapters.get(message.agent_id)
        if adapter is None:
            raise ValueError(
                f'Telegram adapter not configured for agent {message.agent_id}. '
                'Configure agent_channels telegram bot_token first.'
            )
        await adapter.send(message)


class ChannelManager:
    def __init__(self, *, bus: MessageBus, telegram_enabled: bool = True):
        self._bus = bus
        self._telegram_enabled = telegram_enabled
        self._telegram = TelegramChannelService(bus=bus)
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        await register_sender('web', _discard_sender)

        if self._telegram_enabled:
            await self._telegram.start()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._telegram_enabled:
            await self._telegram.stop()

        await unregister_sender('web')

    async def reload_telegram(self) -> None:
        if not self._telegram_enabled:
            return
        await self._telegram.reload()


async def _discard_sender(message: OutboundMessage) -> None:
    # UI/CLI channels read answers directly from storage; no transport send required.
    logger.debug('Discarded outbound message for channel=%s agent=%s', message.channel, message.agent_id)


def _split_message(content: str, max_len: int = 4000) -> list[str]:
    text = content or ''
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        candidate = remaining[:max_len]
        cut = candidate.rfind('\n')
        if cut == -1:
            cut = candidate.rfind(' ')
        if cut <= 0:
            cut = max_len

        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    return chunks


def _markdown_to_telegram_html(text: str) -> str:
    if not text:
        return ''

    code_blocks: list[str] = []
    inline_codes: list[str] = []
    links: list[tuple[str, str]] = []

    def _save_code_block(match: re.Match[str]) -> str:
        code_blocks.append(match.group(1))
        return f'\x00CB{len(code_blocks) - 1}\x00'

    def _save_inline_code(match: re.Match[str]) -> str:
        inline_codes.append(match.group(1))
        return f'\x00IC{len(inline_codes) - 1}\x00'

    def _save_link(match: re.Match[str]) -> str:
        links.append((match.group(1), match.group(2)))
        return f'\x00LK{len(links) - 1}\x00'

    out = text
    out = re.sub(r'```[\w+-]*\n?([\s\S]*?)```', _save_code_block, out)
    out = re.sub(r'`([^`\n]+)`', _save_inline_code, out)
    out = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', _save_link, out)

    out = re.sub(r'^#{1,6}\s+(.+)$', r'\1', out, flags=re.MULTILINE)
    out = re.sub(r'^>\s*(.*)$', r'\1', out, flags=re.MULTILINE)
    out = re.sub(r'^\s*[-*]\s+', '• ', out, flags=re.MULTILINE)

    out = html.escape(out, quote=False)

    out = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', out)
    out = re.sub(r'__(.+?)__', r'<b>\1</b>', out)
    out = re.sub(r'~~(.+?)~~', r'<s>\1</s>', out)
    out = re.sub(r'(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])', r'<i>\1</i>', out)

    for idx, (label, url) in enumerate(links):
        escaped_label = html.escape(label, quote=False)
        escaped_url = html.escape(url, quote=True)
        out = out.replace(f'\x00LK{idx}\x00', f'<a href="{escaped_url}">{escaped_label}</a>')

    for idx, code in enumerate(inline_codes):
        out = out.replace(f'\x00IC{idx}\x00', f'<code>{html.escape(code, quote=False)}</code>')

    for idx, code in enumerate(code_blocks):
        escaped_code = html.escape(code, quote=False)
        out = out.replace(f'\x00CB{idx}\x00', f'<pre><code>{escaped_code}</code></pre>')

    return out


def _telegram_config_from_channel(raw: dict[str, Any]) -> TelegramAgentConfig:
    bot_token = str(raw.get('bot_token') or '').strip()
    proxy_raw = str(raw.get('proxy') or '').strip()
    proxy = proxy_raw or None

    allow_raw = raw.get('allowed_chat_ids')
    allowed_chat_ids: set[str] = set()
    if isinstance(allow_raw, list):
        for item in allow_raw:
            item_text = str(item).strip()
            if item_text:
                allowed_chat_ids.add(item_text)

    poll_timeout = CONFIG.telegram.poll_timeout_seconds
    if isinstance(raw.get('poll_timeout_seconds'), int):
        poll_timeout = max(5, int(raw['poll_timeout_seconds']))

    poll_interval = CONFIG.telegram.poll_interval_seconds
    raw_interval = raw.get('poll_interval_seconds')
    if isinstance(raw_interval, (int, float)):
        poll_interval = max(0.05, float(raw_interval))

    return TelegramAgentConfig(
        bot_token=bot_token,
        allowed_chat_ids=allowed_chat_ids,
        poll_timeout_seconds=poll_timeout,
        poll_interval_seconds=poll_interval,
        proxy=proxy,
    )
