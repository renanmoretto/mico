import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MessageRole = Literal['user', 'assistant', 'tool', 'system']
SCHEMA_VERSION = '3'

_storage_instance: 'SqliteStorage | None' = None


@dataclass(frozen=True)
class AgentRecord:
    id: str
    name: str
    persona: str
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


def _require_storage() -> 'SqliteStorage':
    if _storage_instance is None:
        raise RuntimeError('Storage is not initialized. Call init_storage() before using storage-backed modules.')
    return _storage_instance


def init_storage(db_path: str | None = None) -> 'SqliteStorage':
    if db_path is None:
        from .paths import db_path as get_db_path

        db_path = get_db_path()
    global _storage_instance
    _storage_instance = SqliteStorage.from_path(db_path=db_path)
    return _storage_instance


def is_initialized() -> bool:
    return _storage_instance is not None


def __getattr__(name: str):
    return getattr(_require_storage(), name)


class SqliteStorage:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()

    @classmethod
    def from_path(cls, db_path: str | None = None) -> 'SqliteStorage':
        if db_path is None:
            from .paths import db_path as get_db_path

            db_path = get_db_path()
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            path,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout = 5000;')
        conn.execute('PRAGMA journal_mode = WAL;')
        conn.execute('PRAGMA synchronous = NORMAL;')
        conn.execute('PRAGMA foreign_keys = ON;')

        storage = cls(conn=conn)
        storage._ensure_schema()
        return storage

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute('CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            row = self._conn.execute("SELECT value FROM app_meta WHERE key = 'schema_version' LIMIT 1").fetchone()
            current = str(row['value']) if row is not None else None

            if current != SCHEMA_VERSION:
                # No backward compatibility required: rebuild to keep schema simple.
                self._conn.executescript(
                    """
                    DROP TABLE IF EXISTS config;
                    DROP TABLE IF EXISTS agent_channels;
                    DROP TABLE IF EXISTS memories;
                    DROP TABLE IF EXISTS messages;
                    DROP TABLE IF EXISTS agents;

                    CREATE TABLE agents (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        persona TEXT NOT NULL DEFAULT '',
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
                    """
                )
                self._conn.execute(
                    "INSERT INTO app_meta (key, value) VALUES ('schema_version', ?) "
                    'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                    (SCHEMA_VERSION,),
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._conn.commit()

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
            persona=str(row['persona']),
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

    def _fetch_all(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(query, params).fetchall()

    def _fetch_one(self, query: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(query, params).fetchone()

    def _execute(self, query: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(query, params)
            self._conn.commit()

    def _executemany(self, query: str, params: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(query, params)
            self._conn.commit()

    # Agents

    def create_agent(
        self,
        *,
        name: str,
        persona: str,
        created_at: int,
        runtime: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> str:
        record_id = agent_id or str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO agents (id, name, persona, status, runtime_json, meta_json, created_at, updated_at)
            VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                record_id,
                name,
                persona,
                self._encode_json(runtime),
                self._encode_json(metadata),
                created_at,
                created_at,
            ),
        )
        return record_id

    def upsert_agent(
        self,
        *,
        name: str,
        persona: str,
        updated_at: int,
        runtime: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        row = self._fetch_one('SELECT id FROM agents WHERE name = ? LIMIT 1', (name,))
        if row is None:
            return self.create_agent(
                name=name,
                persona=persona,
                created_at=updated_at,
                runtime=runtime,
                metadata=metadata,
            )

        agent_id = str(row['id'])
        self._execute(
            """
            UPDATE agents
            SET persona = ?, runtime_json = ?, meta_json = ?, updated_at = ?, status = 'active'
            WHERE id = ?
            """,
            (
                persona,
                self._encode_json(runtime),
                self._encode_json(metadata),
                updated_at,
                agent_id,
            ),
        )
        return agent_id

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        row = self._fetch_one(
            """
            SELECT id, name, persona, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE id = ?
            LIMIT 1
            """,
            (agent_id,),
        )
        return self._to_agent_record(row) if row is not None else None

    def find_agent(self, identifier: str) -> AgentRecord | None:
        row = self._fetch_one(
            """
            SELECT id, name, persona, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE id = ? OR name = ?
            LIMIT 1
            """,
            (identifier, identifier),
        )
        return self._to_agent_record(row) if row is not None else None

    def list_agents(self) -> list[AgentRecord]:
        rows = self._fetch_all(
            """
            SELECT id, name, persona, status, runtime_json, meta_json, created_at, updated_at
            FROM agents
            WHERE status != 'deleted'
            ORDER BY created_at ASC
            """
        )
        return [self._to_agent_record(row) for row in rows]

    def update_agent(
        self,
        *,
        agent_id: str,
        updated_at: int,
        name: str | None = None,
        persona: str | None = None,
        status: str | None = None,
        runtime: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        current = self.get_agent(agent_id)
        if current is None:
            return False

        next_name = name if name is not None else current.name
        next_persona = persona if persona is not None else current.persona
        next_status = status if status is not None else current.status
        next_runtime = runtime if runtime is not None else current.runtime
        next_meta = metadata if metadata is not None else current.metadata

        self._execute(
            """
            UPDATE agents
            SET name = ?, persona = ?, status = ?, runtime_json = ?, meta_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                next_name,
                next_persona,
                next_status,
                self._encode_json(next_runtime),
                self._encode_json(next_meta),
                updated_at,
                agent_id,
            ),
        )
        return True

    def delete_agent(self, agent_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute('DELETE FROM agents WHERE id = ?', (agent_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    # Agent channels

    def upsert_agent_channel(
        self,
        *,
        agent_id: str,
        channel: str,
        enabled: bool,
        updated_at: int,
        config: dict[str, Any] | None = None,
    ) -> str:
        new_id = str(uuid.uuid4())
        self._execute(
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
        row = self._fetch_one('SELECT id FROM agent_channels WHERE agent_id = ? AND channel = ?', (agent_id, channel))
        return str(row['id'])

    def get_agent_channel(self, *, agent_id: str, channel: str) -> AgentChannelRecord | None:
        row = self._fetch_one(
            """
            SELECT id, agent_id, channel, enabled, config_json, created_at, updated_at
            FROM agent_channels
            WHERE agent_id = ? AND channel = ?
            LIMIT 1
            """,
            (agent_id, channel),
        )
        return self._to_agent_channel_record(row) if row is not None else None

    def list_agent_channels(self, *, agent_id: str, enabled_only: bool = False) -> list[AgentChannelRecord]:
        extra = ' AND enabled = 1' if enabled_only else ''
        rows = self._fetch_all(
            f'SELECT id, agent_id, channel, enabled, config_json, created_at, updated_at '
            f'FROM agent_channels WHERE agent_id = ?{extra} ORDER BY channel ASC',
            (agent_id,),
        )
        return [self._to_agent_channel_record(row) for row in rows]

    def list_enabled_agent_channels(self, channel: str | None = None) -> list[AgentChannelRecord]:
        extra = ' AND channel = ?' if channel is not None else ''
        params = (channel,) if channel is not None else ()
        rows = self._fetch_all(
            f'SELECT id, agent_id, channel, enabled, config_json, created_at, updated_at '
            f'FROM agent_channels WHERE enabled = 1{extra} ORDER BY agent_id ASC',
            params,
        )
        return [self._to_agent_channel_record(row) for row in rows]

    # App config

    def get_config(self, *, key: str = 'app') -> dict[str, Any] | None:
        row = self._fetch_one(
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

    def upsert_config(self, *, key: str = 'app', config: dict[str, Any]) -> None:
        now = int(time.time())
        self._execute(
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

    def add_message(
        self,
        *,
        agent_id: str,
        role: MessageRole,
        content: str,
        timestamp: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO messages (id, agent_id, timestamp, role, content, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, agent_id, timestamp, role, content, self._encode_json(metadata)),
        )
        return message_id

    def list_messages(self, *, agent_id: str) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT id, agent_id, timestamp, role, content, meta_json
            FROM messages
            WHERE agent_id = ?
            ORDER BY timestamp ASC, rowid ASC
            """,
            (agent_id,),
        )
        return [self._to_message_record(row) for row in rows]

    def list_recent_messages(self, *, agent_id: str, limit: int) -> list[MessageRecord]:
        rows = self._fetch_all(
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

    def list_messages_with_order(self, *, agent_id: str) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT rowid AS insert_order, id, agent_id, timestamp, role, content, meta_json
            FROM messages
            WHERE agent_id = ?
            ORDER BY timestamp ASC, insert_order ASC
            """,
            (agent_id,),
        )
        return [self._to_message_record(row) for row in rows]

    def search_messages(self, *, agent_id: str, query: str, limit: int) -> list[MessageRecord]:
        rows = self._fetch_all(
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

    def delete_messages(self, *, agent_id: str, message_ids: list[str]) -> None:
        if not message_ids:
            return
        self._executemany(
            'DELETE FROM messages WHERE agent_id = ? AND id = ?',
            [(agent_id, message_id) for message_id in message_ids],
        )

    # Memories

    def upsert_memory(
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
        self._execute(
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

    def find_memory(self, *, agent_id: str, identifier: str) -> MemoryRecord | None:
        row = self._fetch_one(
            """
            SELECT id, agent_id, name, summary, content, strength, updated_at, meta_json
            FROM memories
            WHERE agent_id = ? AND (id = ? OR name = ?)
            LIMIT 1
            """,
            (agent_id, identifier, identifier),
        )
        return self._to_memory_record(row) if row is not None else None

    def update_memory(
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
        self._execute(
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

    def delete_memory(self, *, agent_id: str, memory_id: str) -> None:
        self._execute('DELETE FROM memories WHERE agent_id = ? AND id = ?', (agent_id, memory_id))

    def search_memories(self, *, agent_id: str, query: str, limit: int) -> list[MemoryRecord]:
        q = f'%{query.strip()}%'
        rows = self._fetch_all(
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

    def touch_memories(self, *, agent_id: str, memory_ids: list[str], accessed_at: int) -> None:
        if not memory_ids:
            return
        self._executemany(
            """
            UPDATE memories
            SET last_accessed = ?, access_count = access_count + 1
            WHERE agent_id = ? AND id = ?
            """,
            [(accessed_at, agent_id, memory_id) for memory_id in memory_ids],
        )
