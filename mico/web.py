from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from . import agent_config, agents, storage
from . import paths
from .agent import Mico, subscribe_events, unsubscribe_events
from .bus import AgentMessageWorker, MessageBus, OutboundMessageWorker
from .channels import ChannelManager
from . import config as app_config
from .config import CONFIG
from .messages import InboundMessage
from .runtime import get_runtime_manager, reset_runtime_manager
from .scheduler import SchedulerWorker

TEMPLATES_DIR = Path(__file__).parent / 'templates'
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title='Mico Control Panel')
logger = logging.getLogger(__name__)
_mico = Mico()
_storage_initialized = False
_bus: MessageBus | None = None
_agent_worker: AgentMessageWorker | None = None
_outbound_worker: OutboundMessageWorker | None = None
_channel_manager: ChannelManager | None = None
_scheduler_worker: SchedulerWorker | None = None
_pipeline_lock = asyncio.Lock()


async def _ensure_storage() -> None:
    global _storage_initialized
    if _storage_initialized:
        return
    db_path = os.getenv('MICO_DB_PATH') or paths.db_path()
    await storage.init_storage(db_path=db_path)
    _storage_initialized = True


def _telegram_autostart_enabled() -> bool:
    return CONFIG.web.telegram_autostart


async def _start_background_workers() -> None:
    global _bus, _agent_worker, _outbound_worker, _channel_manager, _scheduler_worker

    async with _pipeline_lock:
        if _bus is None:
            _bus = MessageBus()

        _mico.bus = _bus

        if _agent_worker is None:
            _agent_worker = AgentMessageWorker(bus=_bus, mico=_mico)
        if _outbound_worker is None:
            _outbound_worker = OutboundMessageWorker(bus=_bus)
        if _channel_manager is None:
            _channel_manager = ChannelManager(
                bus=_bus,
                telegram_enabled=_telegram_autostart_enabled(),
            )
        if _scheduler_worker is None:
            _scheduler_worker = SchedulerWorker(mico=_mico)

        await _agent_worker.start()
        await _outbound_worker.start()
        await _channel_manager.start()
        await _scheduler_worker.start()

        if not _telegram_autostart_enabled():
            logger.info('Web Telegram autostart is disabled by app config.')


async def _stop_background_workers() -> None:
    global _channel_manager, _outbound_worker, _agent_worker, _bus, _scheduler_worker

    async with _pipeline_lock:
        if _scheduler_worker is not None:
            with suppress(Exception):
                await _scheduler_worker.stop()
            _scheduler_worker = None

        if _channel_manager is not None:
            with suppress(Exception):
                await _channel_manager.stop()
            _channel_manager = None

        if _outbound_worker is not None:
            with suppress(Exception):
                await _outbound_worker.stop()
            _outbound_worker = None

        if _agent_worker is not None:
            with suppress(Exception):
                await _agent_worker.stop()
            _agent_worker = None

        _mico.bus = None
        _bus = None


async def _reload_telegram_service() -> None:
    async with _pipeline_lock:
        if _channel_manager is None:
            return
        await _channel_manager.reload_telegram()


async def _restart_background_workers() -> None:
    await _stop_background_workers()
    await _start_background_workers()


def _require_bus() -> MessageBus:
    if _bus is None:
        raise RuntimeError('Message bus is not initialized.')
    return _bus


@app.on_event('startup')
async def _startup() -> None:
    await _ensure_storage()
    removed = await get_runtime_manager().cleanup_orphaned_containers()
    if removed:
        logger.info('Removed orphaned containers: %s', removed)
    await _start_background_workers()


@app.on_event('shutdown')
async def _shutdown() -> None:
    await _stop_background_workers()


def _notice_params(ok: str | None = None, error: str | None = None) -> str:
    payload: dict[str, str] = {}
    if ok:
        payload['ok'] = ok
    if error:
        payload['error'] = error
    if not payload:
        return ''
    return '?' + urlencode(payload)


def _redirect(target: str, *, ok: str | None = None, error: str | None = None) -> RedirectResponse:
    return RedirectResponse(url=target + _notice_params(ok=ok, error=error), status_code=303)


def _redirect_with_query(
    target: str,
    *,
    query: dict[str, str | None] | None = None,
    ok: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    payload: dict[str, str] = {}
    if query:
        for key, value in query.items():
            if value is None:
                continue
            value_clean = str(value).strip()
            if value_clean:
                payload[key] = value_clean
    if ok:
        payload['ok'] = ok
    if error:
        payload['error'] = error

    url = target
    if payload:
        url += '?' + urlencode(payload)
    return RedirectResponse(url=url, status_code=303)


async def _require_agent(identifier: str) -> storage.AgentRecord:
    row = await storage.get_agent(identifier)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Agent '{identifier}' not found")
    return row


def _truncate(text: str, limit: int = 30_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + '\n...[truncated]'


def _normalize_browse_path(raw: str | None) -> str:
    value = (raw or '.').strip()
    if not value:
        return '.'
    p = PurePosixPath(value)
    normalized = str(p)
    if normalized in {'', '/'}:
        return '.'
    return normalized.lstrip('/')


def _parent_path(path: str) -> str:
    if path in {'', '.'}:
        return '.'
    parent = PurePosixPath(path).parent
    if str(parent) in {'', '.'}:
        return '.'
    return str(parent)


def _normalize_root(raw: str | None) -> str:
    value = str(raw or 'workspace').strip()
    return value or 'workspace'


def _resolve_folder_picker_output(raw: str) -> str | None:
    value = str(raw or '').strip()
    if not value:
        return None
    return str(Path(value).expanduser().resolve())


def _pick_folder_macos() -> str | None:
    result = subprocess.run(
        ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select a folder to attach")'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if '(-128)' in detail:
            return None
        raise RuntimeError(detail or 'macOS folder picker failed.')
    return _resolve_folder_picker_output(result.stdout)


def _pick_folder_windows() -> str | None:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$dialog.Description = 'Select a folder to attach'; "
        "$dialog.ShowNewFolderButton = $false; "
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "Write-Output $dialog.SelectedPath }"
    )
    result = subprocess.run(
        ['powershell', '-NoProfile', '-Command', script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or 'Windows folder picker failed.')
    return _resolve_folder_picker_output(result.stdout)


def _pick_folder_linux() -> str | None:
    for command in (
        ['zenity', '--file-selection', '--directory', '--title=Select a folder to attach'],
        ['kdialog', '--getexistingdirectory', os.path.expanduser('~'), '--title', 'Select a folder to attach'],
    ):
        executable = command[0]
        if shutil.which(executable) is None:
            continue
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return _resolve_folder_picker_output(result.stdout)
        if result.returncode in {1, 130}:
            return None
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f'{executable} folder picker failed.')
    raise RuntimeError('No supported folder picker is available on this Linux machine.')


def _pick_folder_tk() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError('Folder picker is unavailable. Enter the path manually.') from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes('-topmost', True)
    except Exception:
        pass
    try:
        selected = filedialog.askdirectory(mustexist=True, title='Select a folder to attach')
    finally:
        root.destroy()
    return _resolve_folder_picker_output(selected)


def _pick_folder_sync() -> str | None:
    if sys.platform == 'darwin':
        return _pick_folder_macos()
    if sys.platform.startswith('win'):
        return _pick_folder_windows()
    try:
        return _pick_folder_linux()
    except RuntimeError:
        return _pick_folder_tk()


async def _pick_folder() -> str | None:
    return await asyncio.to_thread(_pick_folder_sync)


@app.get('/', response_class=HTMLResponse)
async def home() -> RedirectResponse:
    return RedirectResponse('/agents', status_code=302)


@app.get('/agents', response_class=HTMLResponse)
async def agents_page(request: Request) -> HTMLResponse:
    await _ensure_storage()
    rows = await agents.list_agents()
    channel_map: dict[str, list[agent_config.AgentChannel]] = {}
    for row in rows:
        channel_map[row.id] = await agent_config.list_agent_channels(agent_id=row.id)

    return templates.TemplateResponse(
        request=request,
        name='agents.html',
        context={
            'rows': rows,
            'channels_by_agent': channel_map,
            'ok': request.query_params.get('ok'),
            'error': request.query_params.get('error'),
        },
    )


@app.get('/config', response_class=HTMLResponse)
async def config_page(request: Request) -> HTMLResponse:
    await _ensure_storage()
    payload = await app_config.get_app_config_payload()
    typed = await app_config.get_app_config()
    return templates.TemplateResponse(
        request=request,
        name='config.html',
        context={
            'config_json': json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            'llm_provider': typed.llm.provider,
            'llm_model': typed.llm.model,
            'telegram_defaults': {
                'enabled': typed.telegram.enabled,
                'poll_timeout_seconds': typed.telegram.poll_timeout_seconds,
                'poll_interval_seconds': typed.telegram.poll_interval_seconds,
                'drop_pending_updates': typed.telegram.drop_pending_updates,
            },
            'ok': request.query_params.get('ok'),
            'error': request.query_params.get('error'),
        },
    )


@app.post('/config')
async def config_update(
    config_json: str = Form(...),
) -> RedirectResponse:
    await _ensure_storage()
    raw = config_json.strip()
    if not raw:
        return _redirect('/config', error='Config JSON cannot be empty.')

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _redirect('/config', error=f'Invalid JSON: {exc}')
    if not isinstance(parsed, dict):
        return _redirect('/config', error='Config JSON must be an object at the top level.')

    try:
        await app_config.set_app_config(parsed)
    except Exception as exc:
        return _redirect('/config', error=f'Failed to save config: {exc}')

    reset_runtime_manager()
    try:
        await _restart_background_workers()
    except Exception as exc:
        return _redirect('/config', error=f'Config saved, but workers restart failed: {exc}')
    return _redirect('/config', ok='Config saved and workers restarted.')


@app.post('/agents')
async def create_agent(
    name: str = Form(...),
) -> RedirectResponse:
    await _ensure_storage()
    clean_name = name.strip()
    if not clean_name:
        return _redirect('/agents', error='Agent name is required.')

    try:
        row = await agents.create_agent(name=clean_name)
    except Exception as exc:
        return _redirect('/agents', error=f'Could not create agent: {exc}')
    return _redirect(f'/agents/{row.id}', ok=f"Agent '{row.name}' created.")


@app.get('/agents/{identifier}', response_class=HTMLResponse)
async def agent_detail(request: Request, identifier: str) -> HTMLResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    channels = await agent_config.list_agent_channels(agent_id=row.id)
    telegram = await agent_config.get_agent_channel(agent_id=row.id, channel='telegram')
    recent_messages = await storage.list_recent_messages(agent_id=row.id, limit=30)

    agent_cfg = await agent_config.get_agent_config(row.id)
    agent_config_json = json.dumps(agent_cfg, indent=2, sort_keys=True, ensure_ascii=True)

    telegram_config = telegram.config if telegram is not None else {}
    allow_ids = telegram_config.get('allowed_chat_ids')
    allow_text = ''
    if isinstance(allow_ids, list):
        allow_text = '\n'.join(str(item) for item in allow_ids)

    manager = get_runtime_manager()
    attached_folders = await manager.list_attached_folders(row.id)
    workspace_root = _normalize_root(request.query_params.get('root'))
    workspace_roots = [
        {
            'name': 'workspace',
            'label': 'workspace',
            'path': str(manager.workspace_path(row.id).resolve()),
            'attached': False,
        },
        *[
            {
                'name': folder.name,
                'label': folder.name,
                'path': folder.path,
                'attached': True,
            }
            for folder in attached_folders
        ],
    ]
    workspace_items: list[dict[str, object]] = []
    workspace_error: str | None = None
    file_content: str = ''
    file_error: str | None = None

    browse_path = _normalize_browse_path(request.query_params.get('path'))
    selected_file = request.query_params.get('file')

    try:
        listed = await manager.list_files(agent_id=row.id, path=browse_path, root=workspace_root)
        for item in listed:
            is_dir = item.endswith('/')
            pure = item[:-1] if is_dir else item
            display_name = pure.rsplit('/', 1)[-1]
            if is_dir:
                href = f'/agents/{row.id}?' + urlencode({'tab': 'workspace', 'root': workspace_root, 'path': pure})
            else:
                href = (
                    f'/agents/{row.id}?'
                    + urlencode({'tab': 'workspace', 'root': workspace_root, 'path': browse_path, 'file': pure})
                )
            workspace_items.append(
                {
                    'path': pure,
                    'display': display_name + ('/' if is_dir else ''),
                    'is_dir': is_dir,
                    'href': href,
                }
            )
    except Exception as exc:
        workspace_error = str(exc)

    if selected_file:
        try:
            file_content = await manager.read_file(agent_id=row.id, path=selected_file, root=workspace_root)
            file_content = _truncate(file_content, limit=60_000)
        except Exception as exc:
            file_error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name='agent_detail.html',
        context={
            'agent': row,
            'channels': channels,
            'telegram': telegram,
            'telegram_enabled': bool(telegram.enabled) if telegram is not None else False,
            'telegram_has_token': bool(str(telegram_config.get('bot_token') or '').strip()),
            'telegram_bot_token': str(telegram_config.get('bot_token') or ''),
            'telegram_allowed_chat_ids': allow_text,
            'attached_folders': attached_folders,
            'recent_messages': recent_messages,
            'ok': request.query_params.get('ok'),
            'error': request.query_params.get('error'),
            'workspace_root': workspace_root,
            'workspace_roots': workspace_roots,
            'workspace_path': browse_path,
            'workspace_parent_path': _parent_path(browse_path),
            'workspace_items': workspace_items,
            'workspace_error': workspace_error,
            'selected_file': selected_file,
            'selected_file_content': file_content,
            'selected_file_error': file_error,
            'agent_config_json': agent_config_json,
        },
    )


@app.post('/agents/{identifier}/telegram')
async def update_telegram(
    identifier: str,
    enabled: str | None = Form(default=None),
    bot_token: str = Form(''),
    allowed_chat_ids: str = Form(''),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    is_enabled = enabled == 'on'

    allow_ids: list[str] = []
    if allowed_chat_ids.strip():
        for raw in allowed_chat_ids.replace(',', '\n').splitlines():
            value = raw.strip()
            if value:
                allow_ids.append(value)

    if is_enabled and not bot_token.strip():
        return _redirect(f'/agents/{row.id}', error='Telegram bot token is required when enabled.')

    cfg: dict[str, object] = {
        'bot_token': bot_token.strip(),
        'allowed_chat_ids': allow_ids,
    }
    await agents.configure_channel(
        agent_id=row.id,
        channel='telegram',
        enabled=is_enabled,
        config=cfg,
    )
    try:
        await _reload_telegram_service()
    except Exception as exc:
        return _redirect(f'/agents/{row.id}', error=f'Telegram settings saved, but reload failed: {exc}')
    state = 'enabled' if is_enabled else 'disabled'
    return _redirect(f'/agents/{row.id}', ok=f'Telegram {state}.')


@app.post('/agents/{identifier}/folders/attach')
async def attach_folder(
    identifier: str,
    path: str = Form(''),
    name: str = Form(''),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    manager = get_runtime_manager()
    try:
        folder = await manager.attach_folder(agent_id=row.id, path=path, name=name or None)
    except Exception as exc:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'workspace'}, error=str(exc))
    return _redirect_with_query(
        f'/agents/{row.id}',
        query={'tab': 'workspace', 'root': folder.name, 'path': '.'},
        ok=f"Attached folder '{folder.name}'.",
    )


@app.post('/agents/{identifier}/folders/detach')
async def detach_folder(
    identifier: str,
    name: str = Form(...),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    manager = get_runtime_manager()
    try:
        detached = await manager.detach_folder(agent_id=row.id, name=name)
    except Exception as exc:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'workspace'}, error=str(exc))
    if not detached:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'workspace'}, error=f"Folder '{name}' not found.")
    return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'workspace'}, ok=f"Detached folder '{name}'.")


@app.post('/agents/{identifier}/config')
async def update_agent_config(
    identifier: str,
    config_json: str = Form(...),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    raw = config_json.strip()
    if not raw:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'config'}, error='Config JSON cannot be empty.')

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'config'}, error=f'Invalid JSON: {exc}')
    if not isinstance(parsed, dict):
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'config'}, error='Config must be a JSON object.')

    try:
        await agent_config.set_agent_config(row.id, parsed)
    except Exception as exc:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'config'}, error=f'Failed to save: {exc}')

    try:
        await _reload_telegram_service()
    except Exception as exc:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'config'}, error=f'Saved, but Telegram reload failed: {exc}')

    return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'config'}, ok='Config saved.')


@app.post('/agents/{identifier}/prompt')
async def prompt_agent(
    identifier: str,
    prompt: str = Form(...),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    clean = prompt.strip()
    if not clean:
        return _redirect(f'/agents/{row.id}', error='Prompt cannot be empty.')

    try:
        await _require_bus().publish_inbound(
            InboundMessage(
                agent_id=row.id,
                channel='web',
                sender_id='local-ui',
                chat_id='web-ui',
                content=clean,
                metadata={'source': 'web-control-panel'},
            ),
            wait_response=True,
            timeout_seconds=180.0,
        )
    except Exception as exc:
        return _redirect(f'/agents/{row.id}', error=f'Prompt failed: {exc}')
    return _redirect(f'/agents/{row.id}', ok='Prompt processed. Check recent messages below.')


@app.post('/agents/{identifier}/chat')
async def chat_agent(
    identifier: str,
    message: str = Form(...),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    clean = message.strip()
    if not clean:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'chat'}, error='Message cannot be empty.')

    try:
        await _require_bus().publish_inbound(
            InboundMessage(
                agent_id=row.id,
                channel='web',
                sender_id='local-ui',
                chat_id='web-ui',
                content=clean,
                metadata={'source': 'web-chat'},
            ),
            wait_response=True,
            timeout_seconds=180.0,
        )
    except Exception as exc:
        return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'chat'}, error=f'Chat failed: {exc}')
    return _redirect_with_query(f'/agents/{row.id}', query={'tab': 'chat'}, ok='Message sent.')


@app.post('/api/agents/{identifier}/chat')
async def api_chat_agent(request: Request, identifier: str) -> JSONResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'error': 'Invalid JSON body'}, status_code=400)
    content = str(body.get('message', '')).strip()
    if not content:
        return JSONResponse({'error': 'Message cannot be empty.'}, status_code=400)

    try:
        await _require_bus().publish_inbound(
            InboundMessage(
                agent_id=row.id,
                channel='web',
                sender_id='local-ui',
                chat_id='web-ui',
                content=content,
                metadata={'source': 'web-chat'},
            ),
            wait_response=True,
            timeout_seconds=180.0,
        )
    except Exception as exc:
        return JSONResponse({'error': f'Chat failed: {exc}'}, status_code=500)

    messages = await storage.list_recent_messages(agent_id=row.id, limit=50)
    return JSONResponse(
        {
            'success': True,
            'messages': [_message_to_dict(m) for m in messages],
        }
    )


def _message_to_dict(m: storage.MessageRecord) -> dict:
    d: dict = {'id': m.id, 'role': m.role, 'timestamp': m.timestamp}
    if m.content is not None:
        d['content'] = m.content
    if m.tool_call_id is not None:
        d['tool_call_id'] = m.tool_call_id
    if m.tool_calls is not None:
        d['tool_calls'] = m.tool_calls
    if m.metadata:
        d['metadata'] = m.metadata
    return d


@app.get('/api/agents/{identifier}/messages')
async def api_get_messages(identifier: str, since: int = 0, limit: int = 50) -> JSONResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    messages = await storage.list_recent_messages(agent_id=row.id, limit=limit)
    filtered = [m for m in messages if m.timestamp > since]
    return JSONResponse(
        {
            'messages': [_message_to_dict(m) for m in filtered],
        }
    )


@app.post('/api/folders/pick')
async def api_pick_folder() -> JSONResponse:
    try:
        selected = await _pick_folder()
    except Exception as exc:
        return JSONResponse({'error': str(exc)}, status_code=500)
    if not selected:
        return JSONResponse({'cancelled': True})
    return JSONResponse({'path': selected})


@app.get('/api/agents/{identifier}/events')
async def agent_events(identifier: str) -> StreamingResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    q = subscribe_events(row.id)

    async def event_stream():
        import time as _time
        deadline = _time.time() + 300
        try:
            while _time.time() < deadline:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            unsubscribe_events(row.id, q)

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.post('/agents/{identifier}/workspace/open')
async def workspace_open(
    identifier: str,
    path: str = Form('.'),
    root: str = Form('workspace'),
) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    browse = _normalize_browse_path(path)
    return _redirect_with_query(
        f'/agents/{row.id}',
        query={'tab': 'workspace', 'root': _normalize_root(root), 'path': browse},
    )


@app.post('/agents/{identifier}/delete')
async def delete_agent(identifier: str) -> RedirectResponse:
    await _ensure_storage()
    row = await _require_agent(identifier)
    try:
        await get_runtime_manager().delete_agent_runtime(row.id)
    except Exception:
        # Runtime cleanup failures shouldn't block data deletion.
        pass

    deleted = await agents.delete_agent(row.id)
    if not deleted:
        return _redirect('/agents', error=f"Could not delete agent '{row.name}'.")
    return _redirect('/agents', ok=f"Agent '{row.name}' deleted.")
