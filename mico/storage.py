import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite

MessageRole = Literal['user', 'assistant', 'tool', 'system']
SCHEMA_VERSION = '8'

_storage_instance: 'SqliteStorage | None' = None


@dataclass(frozen=True)
class AgentRecord:
    id: str
    name: str
    status: str
    runtime: dict[str, Any]
    metadata: dict[str, Any]
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class MessageRecord:
    id: str
    agent_id: str
    timestamp: int
    role: str
    content: str | None
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]] | None
    metadata: dict[str, Any]
    insert_order: int | None = None


@dataclass(frozen=True)
class ScheduledJobRecord:
    id: str
    agent_id: str
    description: str
    instruction: str
    job_type: str  # 'once' | 'recurring'
    cron_expr: str | None
    next_run_at: int
    last_run_at: int | None
    status: str  # 'active' | 'paused' | 'completed'
    created_at: int
    updated_at: int


def _require_storage() -> 'SqliteStorage':
    if _storage_instance is None:
        raise RuntimeError('Storage is not initialized. Call init_storage() before using storage-backed modules.')
    return _storage_instance


async def init_storage(db_path: str | None = None) -> 'SqliteStorage':
    if db_path is None:
        from .paths import db_path as get_db_path

        db_path = get_db_path()
    global _storage_instance
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    conn.row_factory = sqlite3.Row
    await conn.execute('PRAGMA journal_mode=WAL')
    await conn.execute('PRAGMA busy_timeout = 5000;')
    await conn.execute('PRAGMA synchronous = NORMAL;')
    await conn.execute('PRAGMA foreign_keys = ON;')
    storage = SqliteStorage(conn=conn)
    await storage._ensure_schema()
    _storage_instance = storage
    return _storage_instance


def is_initialized() -> bool:
    return _storage_instance is not None


def __getattr__(name: str):
    return getattr(_require_storage(), name)


class SqliteStorage:
    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn
        self._lock = asyncio.Lock()

    async def _ensure_schema(self) -> None:
        async with self._lock:
            async with self._conn.execute(
                'CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
            ) as cursor:
                pass
            async with self._conn.execute(
                "SELECT value FROM app_meta WHERE key = 'schema_version' LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            current = str(row['value']) if row is not None else None

            if current != SCHEMA_VERSION:
                # Schema changes rebuild the local DB to keep storage simple.
                await self._conn.executescript(
                    """
                    DROP TABLE IF EXISTS scheduled_jobs;
                    DROP TABLE IF EXISTS config;
                    DROP TABLE IF EXISTS agent_channels;
                    DROP TABLE IF EXISTS memories;
                    DROP TABLE IF EXISTS messages;
                    DROP TABLE IF EXISTS agents;

                    CREATE TABLE agents (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL DEFAULT 'active',
                        runtime_json TEXT NOT NULL DEFAULT '{}',
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        CHECK (status IN ('active', 'paused', 'deleted'))
                    );

                    CREATE TABLE config (
                        key TEXT PRIMARY KEY,
                        config_json TEXT NOT NULL DEFAULT '{}',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    );

                    CREATE TABLE messages (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT,
                        tool_call_id TEXT,
                        tool_calls_json TEXT,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
                        CHECK (role IN ('user', 'assistant', 'tool', 'system'))
                    );

                    CREATE INDEX idx_messages_agent_ts ON messages(agent_id, timestamp);

                    CREATE TABLE scheduled_jobs (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        description TEXT NOT NULL,
                        instruction TEXT NOT NULL,
                        job_type TEXT NOT NULL,
                        cron_expr TEXT,
                        next_run_at INTEGER NOT NULL,
                        last_run_at INTEGER,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
                        CHECK (job_type IN ('once', 'recurring')),
                        CHECK (status IN ('active', 'paused', 'completed'))
                    );
                    CREATE INDEX idx_scheduled_jobs_due ON scheduled_jobs(next_run_at)
                        WHERE status = 'active';
                    """
                )
                await self._conn.execute(
                    "INSERT INTO app_meta (key, value) VALUES ('schema_version', ?) "
                    'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                    (SCHEMA_VERSION,),
                )
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            await self._conn.commit()

    @staticmethod
    def _encode_json(value: dict[str, Any] | None) -> str:
        return json.dumps(value or {}, ensure_ascii=True, separators=(',', ':'))

    @staticmethod
    def _encode_json_value(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=True, separators=(',', ':'))

    @staticmethod
    def _decode_json(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        text = str(value).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _decode_json_list(value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        return [item for item in parsed if isinstance(item, dict)] or None

    @staticmethod
    def _to_agent_record(row: sqlite3.Row) -> AgentRecord:
        return AgentRecord(
            id=str(row['id']),
            name=str(row['name']),
            status=str(row['status']),
            runtime=SqliteStorage._decode_json(row['runtime_json']),
            metadata=SqliteStorage._decode_json(row['meta_json']),
            created_at=int(row['created_at']),
            updated_at=int(row['updated_at']),
        )

    @staticmethod
    def _to_message_record(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=str(row['id']),
            agent_id=str(row['agent_id']),
            timestamp=int(row['timestamp']),
            role=str(row['role']),
            content=str(row['content']) if row['content'] is not None else None,
            tool_call_id=str(row['tool_call_id']) if row['tool_call_id'] is not None else None,
            tool_calls=SqliteStorage._decode_json_list(row['tool_calls_json']),
            metadata=SqliteStorage._decode_json(row['meta_json']),
            insert_order=int(row['insert_order']) if 'insert_order' in row.keys() else None,
        )

    @staticmethod
    def _to_scheduled_job_record(row: sqlite3.Row) -> ScheduledJobRecord:
        return ScheduledJobRecord(
            id=str(row['id']),
            agent_id=str(row['agent_id']),
            description=str(row['description']),
            instruction=str(row['instruction']),
            job_type=str(row['job_type']),
            cron_expr=str(row['cron_expr']) if row['cron_expr'] is not None else None,
            next_run_at=int(row['next_run_at']),
            last_run_at=int(row['last_run_at']) if row['last_run_at'] is not None else None,
            status=str(row['status']),
            created_at=int(row['created_at']),
            updated_at=int(row['updated_at']),
        )

    async def _fetch_all(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        async with self._lock:
            async with self._conn.execute(query, params) as cursor:
                return await cursor.fetchall()

    async def _fetch_one(self, query: str, params: tuple = ()) -> sqlite3.Row | None:
        async with self._lock:
            async with self._conn.execute(query, params) as cursor:
                return await cursor.fetchone()

    async def _execute(self, query: str, params: tuple = ()) -> None:
        async with self._lock:
            await self._conn.execute(query, params)
            await self._conn.commit()

    async def _executemany(self, query: str, params: list[tuple]) -> None:
        async with self._lock:
            await self._conn.executemany(query, params)
            await self._conn.commit()

    # Agents

    async def create_agent(
        self,
        *,
        name: str,
        created_at: int,
        runtime: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> str:
        record_id = agent_id or str(uuid.uuid4())
        await self._execute(
            """
            INSERT INTO agents (id, name, status, runtime_json, meta_json, created_at, updated_at)
            VALUES (?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                record_id,
                name,
                self._encode_json(runtime),
                self._encode_json(metadata),
                created_at,
                created_at,
            ),
        )
        return record_id

    async def upsert_agent(
        self,
        *,
        name: str,
        updated_at: int,
        runtime: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        row = await self._fetch_one('SELECT id FROM agents WHERE name = ? LIMIT 1', (name,))
        if row is None:
            return await self.create_agent(
                name=name,
                created_at=updated_at,
                runtime=runtime,
                metadata=metadata,
            )

        agent_id = str(row['id'])
        await self._execute(
            """
            UPDATE agents
            SET runtime_json = ?, meta_json = ?, updated_at = ?, status = 'active'
            WHERE id = ?
            """,
            (
                self._encode_json(runtime),
                self._encode_json(metadata),
                updated_at,
                agent_id,
            ),
        )
        return agent_id

    async def get_agent(self, agent_id: str) -> AgentRecord | None:
        row = await self._fetch_one(
            """
            SELECT id, name, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE id = ?
            LIMIT 1
            """,
            (agent_id,),
        )
        return self._to_agent_record(row) if row is not None else None

    async def get_agent_by_name(self, name: str) -> AgentRecord | None:
        row = await self._fetch_one(
            """
            SELECT id, name, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE name = ?
            LIMIT 1
            """,
            (name,),
        )
        return self._to_agent_record(row) if row is not None else None

    async def list_agents(self) -> list[AgentRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, name, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE status != 'deleted'
            ORDER BY created_at ASC
            """
        )
        return [self._to_agent_record(row) for row in rows]

    async def update_agent(
        self,
        *,
        agent_id: str,
        updated_at: int,
        name: str | None = None,
        status: str | None = None,
        runtime: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        current = await self.get_agent(agent_id)
        if current is None:
            return False

        next_name = name if name is not None else current.name
        next_status = status if status is not None else current.status
        next_runtime = runtime if runtime is not None else current.runtime
        next_meta = metadata if metadata is not None else current.metadata

        await self._execute(
            """
            UPDATE agents
            SET name = ?, status = ?, runtime_json = ?, meta_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                next_name,
                next_status,
                self._encode_json(next_runtime),
                self._encode_json(next_meta),
                updated_at,
                agent_id,
            ),
        )
        return True

    async def delete_agent(self, agent_id: str) -> bool:
        async with self._lock:
            cursor = await self._conn.execute('DELETE FROM agents WHERE id = ?', (agent_id,))
            await self._conn.commit()
            return cursor.rowcount > 0

    # App config

    async def get_config(self, *, key: str = 'app') -> dict[str, Any] | None:
        row = await self._fetch_one(
            """
            SELECT config_json
            FROM config
            WHERE key = ?
            LIMIT 1
            """,
            (key,),
        )
        if row is None:
            return None
        return self._decode_json(row['config_json'])

    async def upsert_config(self, *, key: str = 'app', config: dict[str, Any]) -> None:
        now = int(time.time())
        await self._execute(
            """
            INSERT INTO config (key, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = excluded.updated_at
            """,
            (key, self._encode_json(config), now, now),
        )

    # Messages

    async def add_message(
        self,
        *,
        agent_id: str,
        role: MessageRole,
        content: str | None,
        timestamp: int,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        await self._execute(
            """
            INSERT INTO messages (
                id,
                agent_id,
                timestamp,
                role,
                content,
                tool_call_id,
                tool_calls_json,
                meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                agent_id,
                timestamp,
                role,
                content,
                tool_call_id,
                self._encode_json_value(tool_calls),
                self._encode_json(metadata),
            ),
        )
        return message_id

    async def list_messages(self, *, agent_id: str) -> list[MessageRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, timestamp, role, content, tool_call_id, tool_calls_json, meta_json
            FROM messages
            WHERE agent_id = ?
            ORDER BY timestamp ASC, rowid ASC
            """,
            (agent_id,),
        )
        return [self._to_message_record(row) for row in rows]

    async def list_recent_messages(self, *, agent_id: str, limit: int) -> list[MessageRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, timestamp, role, content, tool_call_id, tool_calls_json, meta_json
            FROM (
                SELECT rowid AS insert_order, id, agent_id, timestamp, role, content, tool_call_id, tool_calls_json, meta_json
                FROM messages
                WHERE agent_id = ?
                ORDER BY timestamp DESC, insert_order DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC, insert_order ASC
            """,
            (agent_id, limit),
        )
        return [self._to_message_record(row) for row in rows]

    async def list_messages_with_order(self, *, agent_id: str) -> list[MessageRecord]:
        rows = await self._fetch_all(
            """
            SELECT rowid AS insert_order, id, agent_id, timestamp, role, content, tool_call_id, tool_calls_json, meta_json
            FROM messages
            WHERE agent_id = ?
            ORDER BY timestamp ASC, insert_order ASC
            """,
            (agent_id,),
        )
        return [self._to_message_record(row) for row in rows]

    async def search_messages(self, *, agent_id: str, query: str, limit: int) -> list[MessageRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, timestamp, role, content, tool_call_id, tool_calls_json, meta_json
            FROM messages
            WHERE agent_id = ?
              AND (
                COALESCE(content, '') LIKE ?
                OR COALESCE(tool_call_id, '') LIKE ?
                OR COALESCE(tool_calls_json, '') LIKE ?
              )
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
            """,
            (
                agent_id,
                f'%{query.strip()}%',
                f'%{query.strip()}%',
                f'%{query.strip()}%',
                limit,
            ),
        )
        return [self._to_message_record(row) for row in rows]

    async def delete_messages(self, *, agent_id: str, message_ids: list[str]) -> None:
        if not message_ids:
            return
        await self._executemany(
            'DELETE FROM messages WHERE agent_id = ? AND id = ?',
            [(agent_id, message_id) for message_id in message_ids],
        )

    # Scheduled jobs

    async def create_scheduled_job(
        self,
        *,
        agent_id: str,
        description: str,
        instruction: str,
        job_type: str,
        cron_expr: str | None,
        next_run_at: int,
        created_at: int,
    ) -> str:
        job_id = str(uuid.uuid4())
        await self._execute(
            """
            INSERT INTO scheduled_jobs (id, agent_id, description, instruction, job_type, cron_expr, next_run_at, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (job_id, agent_id, description, instruction, job_type, cron_expr, next_run_at, created_at, created_at),
        )
        return job_id

    async def get_due_jobs(self, now: int) -> list[ScheduledJobRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, description, instruction, job_type, cron_expr, next_run_at, last_run_at, status, created_at, updated_at
            FROM scheduled_jobs
            WHERE status = 'active' AND next_run_at <= ?
            ORDER BY next_run_at ASC
            """,
            (now,),
        )
        return [self._to_scheduled_job_record(row) for row in rows]

    async def update_job_after_run(self, *, job_id: str, next_run_at: int | None, status: str, last_run_at: int) -> None:
        await self._execute(
            """
            UPDATE scheduled_jobs
            SET next_run_at = ?, status = ?, last_run_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (next_run_at, status, last_run_at, last_run_at, job_id),
        )

    async def list_scheduled_jobs(self, *, agent_id: str) -> list[ScheduledJobRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, description, instruction, job_type, cron_expr, next_run_at, last_run_at, status, created_at, updated_at
            FROM scheduled_jobs
            WHERE agent_id = ? AND status = 'active'
            ORDER BY next_run_at ASC
            """,
            (agent_id,),
        )
        return [self._to_scheduled_job_record(row) for row in rows]

    async def delete_scheduled_job(self, *, agent_id: str, job_id: str) -> bool:
        async with self._lock:
            cursor = await self._conn.execute('DELETE FROM scheduled_jobs WHERE id = ? AND agent_id = ?', (job_id, agent_id))
            await self._conn.commit()
            return cursor.rowcount > 0
