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
SCHEMA_VERSION = '5'

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
class AgentChannelRecord:
    id: str
    agent_id: str
    channel: str
    enabled: bool
    config: dict[str, Any]
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class MessageRecord:
    id: str
    agent_id: str
    timestamp: int
    role: str
    content: str
    metadata: dict[str, Any]
    insert_order: int | None = None


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    agent_id: str
    name: str
    summary: str
    content: str
    strength: int
    updated_at: int
    metadata: dict[str, Any]


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
                # No backward compatibility required: rebuild to keep schema simple.
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

                    CREATE TABLE agent_channels (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        config_json TEXT NOT NULL DEFAULT '{}',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
                        UNIQUE(agent_id, channel)
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
                        content TEXT NOT NULL,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
                        CHECK (role IN ('user', 'assistant', 'tool', 'system'))
                    );

                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        last_accessed INTEGER,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        strength INTEGER NOT NULL,
                        summary TEXT NOT NULL,
                        content TEXT NOT NULL,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
                        CHECK (strength >= 0 AND strength <= 5),
                        UNIQUE(agent_id, name)
                    );

                    CREATE INDEX idx_messages_agent_ts ON messages(agent_id, timestamp);
                    CREATE INDEX idx_memories_agent_updated ON memories(agent_id, updated_at);
                    CREATE INDEX idx_memories_agent_strength ON memories(agent_id, strength);
                    CREATE INDEX idx_agent_channels_lookup ON agent_channels(agent_id, channel);

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
    def _to_agent_channel_record(row: sqlite3.Row) -> AgentChannelRecord:
        return AgentChannelRecord(
            id=str(row['id']),
            agent_id=str(row['agent_id']),
            channel=str(row['channel']),
            enabled=bool(int(row['enabled'])),
            config=SqliteStorage._decode_json(row['config_json']),
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
            content=str(row['content']),
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

    @staticmethod
    def _to_memory_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=str(row['id']),
            agent_id=str(row['agent_id']),
            name=str(row['name']),
            summary=str(row['summary']),
            content=str(row['content']),
            strength=int(row['strength']),
            updated_at=int(row['updated_at']),
            metadata=SqliteStorage._decode_json(row['meta_json']),
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

    async def find_agent(self, identifier: str) -> AgentRecord | None:
        row = await self._fetch_one(
            """
            SELECT id, name, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE id = ? OR name = ?
            LIMIT 1
            """,
            (identifier, identifier),
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

    # Agent channels

    async def upsert_agent_channel(
        self,
        *,
        agent_id: str,
        channel: str,
        enabled: bool,
        updated_at: int,
        config: dict[str, Any] | None = None,
    ) -> str:
        new_id = str(uuid.uuid4())
        await self._execute(
            """
            INSERT INTO agent_channels (id, agent_id, channel, enabled, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, channel) DO UPDATE SET
                enabled = excluded.enabled,
                config_json = excluded.config_json,
                updated_at = excluded.updated_at
            """,
            (new_id, agent_id, channel, 1 if enabled else 0, self._encode_json(config), updated_at, updated_at),
        )
        row = await self._fetch_one('SELECT id FROM agent_channels WHERE agent_id = ? AND channel = ?', (agent_id, channel))
        return str(row['id'])

    async def get_agent_channel(self, *, agent_id: str, channel: str) -> AgentChannelRecord | None:
        row = await self._fetch_one(
            """
            SELECT id, agent_id, channel, enabled, config_json, created_at, updated_at
            FROM agent_channels
            WHERE agent_id = ? AND channel = ?
            LIMIT 1
            """,
            (agent_id, channel),
        )
        return self._to_agent_channel_record(row) if row is not None else None

    async def list_agent_channels(self, *, agent_id: str, enabled_only: bool = False) -> list[AgentChannelRecord]:
        extra = ' AND enabled = 1' if enabled_only else ''
        rows = await self._fetch_all(
            f'SELECT id, agent_id, channel, enabled, config_json, created_at, updated_at '
            f'FROM agent_channels WHERE agent_id = ?{extra} ORDER BY channel ASC',
            (agent_id,),
        )
        return [self._to_agent_channel_record(row) for row in rows]

    async def list_enabled_agent_channels(self, channel: str | None = None) -> list[AgentChannelRecord]:
        extra = ' AND channel = ?' if channel is not None else ''
        params = (channel,) if channel is not None else ()
        rows = await self._fetch_all(
            f'SELECT id, agent_id, channel, enabled, config_json, created_at, updated_at '
            f'FROM agent_channels WHERE enabled = 1{extra} ORDER BY agent_id ASC',
            params,
        )
        return [self._to_agent_channel_record(row) for row in rows]

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
        content: str,
        timestamp: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        await self._execute(
            """
            INSERT INTO messages (id, agent_id, timestamp, role, content, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, agent_id, timestamp, role, content, self._encode_json(metadata)),
        )
        return message_id

    async def list_messages(self, *, agent_id: str) -> list[MessageRecord]:
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, timestamp, role, content, meta_json
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
            SELECT id, agent_id, timestamp, role, content, meta_json
            FROM (
                SELECT rowid AS insert_order, id, agent_id, timestamp, role, content, meta_json
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
            SELECT rowid AS insert_order, id, agent_id, timestamp, role, content, meta_json
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
            SELECT id, agent_id, timestamp, role, content, meta_json
            FROM messages
            WHERE agent_id = ? AND content LIKE ?
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
            """,
            (agent_id, f'%{query.strip()}%', limit),
        )
        return [self._to_message_record(row) for row in rows]

    async def delete_messages(self, *, agent_id: str, message_ids: list[str]) -> None:
        if not message_ids:
            return
        await self._executemany(
            'DELETE FROM messages WHERE agent_id = ? AND id = ?',
            [(agent_id, message_id) for message_id in message_ids],
        )

    # Memories

    async def upsert_memory(
        self,
        *,
        agent_id: str,
        name: str,
        summary: str,
        content: str,
        strength: int,
        updated_at: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        memory_id = str(uuid.uuid4())
        await self._execute(
            """
            INSERT INTO memories (
                id, agent_id, name, created_at, updated_at, last_accessed, access_count,
                strength, summary, content, meta_json
            )
            VALUES (?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?)
            ON CONFLICT(agent_id, name) DO UPDATE SET
                summary = excluded.summary,
                content = excluded.content,
                strength = excluded.strength,
                updated_at = excluded.updated_at,
                meta_json = excluded.meta_json
            """,
            (
                memory_id,
                agent_id,
                name,
                updated_at,
                updated_at,
                strength,
                summary,
                content,
                self._encode_json(metadata),
            ),
        )

    async def find_memory(self, *, agent_id: str, identifier: str) -> MemoryRecord | None:
        row = await self._fetch_one(
            """
            SELECT id, agent_id, name, summary, content, strength, updated_at, meta_json
            FROM memories
            WHERE agent_id = ? AND (id = ? OR name = ?)
            LIMIT 1
            """,
            (agent_id, identifier, identifier),
        )
        return self._to_memory_record(row) if row is not None else None

    async def update_memory(
        self,
        *,
        agent_id: str,
        memory_id: str,
        name: str,
        summary: str,
        content: str,
        strength: int,
        updated_at: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._execute(
            """
            UPDATE memories
            SET name = ?, summary = ?, content = ?, strength = ?, updated_at = ?, meta_json = ?
            WHERE agent_id = ? AND id = ?
            """,
            (
                name,
                summary,
                content,
                strength,
                updated_at,
                self._encode_json(metadata),
                agent_id,
                memory_id,
            ),
        )

    async def delete_memory(self, *, agent_id: str, memory_id: str) -> None:
        await self._execute('DELETE FROM memories WHERE agent_id = ? AND id = ?', (agent_id, memory_id))

    async def search_memories(self, *, agent_id: str, query: str, limit: int) -> list[MemoryRecord]:
        q = f'%{query.strip()}%'
        rows = await self._fetch_all(
            """
            SELECT id, agent_id, name, strength, summary, content, updated_at, meta_json
            FROM memories
            WHERE agent_id = ? AND (name LIKE ? OR summary LIKE ? OR content LIKE ?)
            ORDER BY strength DESC, updated_at DESC
            LIMIT ?
            """,
            (agent_id, q, q, q, limit),
        )
        return [self._to_memory_record(row) for row in rows]

    async def touch_memories(self, *, agent_id: str, memory_ids: list[str], accessed_at: int) -> None:
        if not memory_ids:
            return
        await self._executemany(
            """
            UPDATE memories
            SET last_accessed = ?, access_count = access_count + 1
            WHERE agent_id = ? AND id = ?
            """,
            [(accessed_at, agent_id, memory_id) for memory_id in memory_ids],
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
