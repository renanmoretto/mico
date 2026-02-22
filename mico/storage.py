import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MessageRole = Literal['user', 'assistant', 'tool', 'system']
_storage_instance: 'SqliteStorage | None' = None


@dataclass(frozen=True)
class MessageRecord:
    id: str
    timestamp: int
    role: str
    content: str
    insert_order: int | None = None


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    name: str
    summary: str
    content: str
    strength: int
    updated_at: int


@dataclass(frozen=True)
class CronRecord:
    id: str
    name: str
    prompt: str
    run_at: int
    status: str
    created_at: int
    updated_at: int
    last_error: str | None = None
    completed_at: int | None = None


def _require_storage() -> 'SqliteStorage':
    if _storage_instance is None:
        raise RuntimeError('Storage is not initialized. Call init_storage() before using storage-backed modules.')
    return _storage_instance


def init_storage(db_path: str = '.db/mico.db') -> 'SqliteStorage':
    global _storage_instance
    _storage_instance = SqliteStorage.from_path(db_path=db_path)
    return _storage_instance


def __getattr__(name: str):
    return getattr(_require_storage(), name)


class SqliteStorage:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()

    @classmethod
    def from_path(cls, db_path: str = '.db/mico.db') -> 'SqliteStorage':
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
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_accessed INTEGER,
                access_count INTEGER NOT NULL DEFAULT 0,
                strength INTEGER NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                CHECK (strength >= 0 AND strength <= 5)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                run_at INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_error TEXT,
                completed_at INTEGER,
                CHECK (status IN ('pending', 'running', 'done', 'error'))
            );
            """
        )
        conn.commit()
        return cls(conn=conn)

    @staticmethod
    def _to_message_record(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=str(row['id']),
            timestamp=int(row['timestamp']),
            role=str(row['role']),
            content=str(row['content']),
            insert_order=int(row['insert_order']) if 'insert_order' in row.keys() else None,
        )

    @staticmethod
    def _to_memory_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=str(row['id']),
            name=str(row['name']),
            summary=str(row['summary']),
            content=str(row['content']),
            strength=int(row['strength']),
            updated_at=int(row['updated_at']),
        )

    @staticmethod
    def _to_cron_record(row: sqlite3.Row) -> CronRecord:
        return CronRecord(
            id=str(row['id']),
            name=str(row['name']),
            prompt=str(row['prompt']),
            run_at=int(row['run_at']),
            status=str(row['status']),
            created_at=int(row['created_at']),
            updated_at=int(row['updated_at']),
            last_error=str(row['last_error']) if row['last_error'] is not None else None,
            completed_at=int(row['completed_at']) if row['completed_at'] is not None else None,
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

    def add_message(self, role: MessageRole, content: str, timestamp: int) -> str:
        message_id = str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO messages (id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (message_id, timestamp, role, content),
        )
        return message_id

    def list_messages(self) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT id, timestamp, role, content
            FROM messages
            ORDER BY timestamp ASC, rowid ASC
            """
        )
        return [self._to_message_record(row) for row in rows]

    def list_recent_messages(self, limit: int) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT id, timestamp, role, content
            FROM (
                SELECT rowid AS insert_order, id, timestamp, role, content
                FROM messages
                ORDER BY timestamp DESC, insert_order DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC, insert_order ASC
            """,
            (limit,),
        )
        return [self._to_message_record(row) for row in rows]

    def list_messages_with_order(self) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT rowid AS insert_order, id, timestamp, role, content
            FROM messages
            ORDER BY timestamp ASC, insert_order ASC
            """
        )
        return [self._to_message_record(row) for row in rows]

    def search_messages(self, query: str, limit: int) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT id, timestamp, role, content
            FROM messages
            WHERE content LIKE ?
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
            """,
            (f'%{query.strip()}%', limit),
        )
        return [self._to_message_record(row) for row in rows]

    def delete_messages(self, message_ids: list[str]) -> None:
        if not message_ids:
            return
        self._executemany('DELETE FROM messages WHERE id = ?', [(message_id,) for message_id in message_ids])

    def upsert_memory(
        self,
        name: str,
        summary: str,
        content: str,
        strength: int,
        updated_at: int,
    ) -> None:
        memory_id = str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO memories (id, name, created_at, updated_at, last_accessed, access_count, strength, summary, content)
            VALUES (?, ?, ?, ?, NULL, 0, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                summary = excluded.summary,
                content = excluded.content,
                strength = excluded.strength,
                updated_at = excluded.updated_at
            """,
            (memory_id, name, updated_at, updated_at, strength, summary, content),
        )

    def find_memory(self, identifier: str) -> MemoryRecord | None:
        row = self._fetch_one(
            """
            SELECT id, name, summary, content, strength, updated_at
            FROM memories
            WHERE id = ? OR name = ?
            LIMIT 1
            """,
            (identifier, identifier),
        )
        return self._to_memory_record(row) if row is not None else None

    def update_memory(
        self,
        memory_id: str,
        name: str,
        summary: str,
        content: str,
        strength: int,
        updated_at: int,
    ) -> None:
        self._execute(
            """
            UPDATE memories
            SET name = ?, summary = ?, content = ?, strength = ?, updated_at = ?
            WHERE id = ?
            """,
            (name, summary, content, strength, updated_at, memory_id),
        )

    def delete_memory(self, memory_id: str) -> None:
        self._execute('DELETE FROM memories WHERE id = ?', (memory_id,))

    def search_memories(self, query: str, limit: int) -> list[MemoryRecord]:
        q = f'%{query.strip()}%'
        rows = self._fetch_all(
            """
            SELECT id, name, strength, summary, content, updated_at
            FROM memories
            WHERE name LIKE ? OR summary LIKE ? OR content LIKE ?
            ORDER BY strength DESC, updated_at DESC
            LIMIT ?
            """,
            (q, q, q, limit),
        )
        return [self._to_memory_record(row) for row in rows]

    def touch_memories(self, memory_ids: list[str], accessed_at: int) -> None:
        if not memory_ids:
            return
        self._executemany(
            """
            UPDATE memories
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
            """,
            [(accessed_at, memory_id) for memory_id in memory_ids],
        )

    def create_cron(self, name: str, prompt: str, run_at: int, created_at: int) -> str:
        cron_id = str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO crons (id, name, prompt, run_at, status, created_at, updated_at, last_error, completed_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, NULL, NULL)
            """,
            (cron_id, name, prompt, run_at, created_at, created_at),
        )
        return cron_id

    def list_crons(self, limit: int, include_done: bool = False) -> list[CronRecord]:
        if include_done:
            rows = self._fetch_all(
                """
                SELECT id, name, prompt, run_at, status, created_at, updated_at, last_error, completed_at
                FROM crons
                ORDER BY run_at ASC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            rows = self._fetch_all(
                """
                SELECT id, name, prompt, run_at, status, created_at, updated_at, last_error, completed_at
                FROM crons
                WHERE status != 'done'
                ORDER BY run_at ASC
                LIMIT ?
                """,
                (limit,),
            )
        return [self._to_cron_record(row) for row in rows]

    def delete_cron(self, cron_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute('DELETE FROM crons WHERE id = ?', (cron_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def claim_due_crons(self, now: int, limit: int = 10) -> list[CronRecord]:
        rows = self._fetch_all(
            """
            SELECT id, name, prompt, run_at, status, created_at, updated_at, last_error, completed_at
            FROM crons
            WHERE status = 'pending' AND run_at <= ?
            ORDER BY run_at ASC
            LIMIT ?
            """,
            (now, limit),
        )
        if not rows:
            return []

        claimed = [self._to_cron_record(row) for row in rows]
        self._executemany(
            """
            UPDATE crons
            SET status = 'running', updated_at = ?, last_error = NULL
            WHERE id = ?
            """,
            [(now, row.id) for row in claimed],
        )
        return [
            CronRecord(
                id=row.id,
                name=row.name,
                prompt=row.prompt,
                run_at=row.run_at,
                status='running',
                created_at=row.created_at,
                updated_at=now,
                last_error=None,
                completed_at=row.completed_at,
            )
            for row in claimed
        ]

    def mark_cron_done(self, cron_id: str, finished_at: int) -> None:
        self._execute(
            """
            UPDATE crons
            SET status = 'done', updated_at = ?, completed_at = ?, last_error = NULL
            WHERE id = ?
            """,
            (finished_at, finished_at, cron_id),
        )

    def mark_cron_error(self, cron_id: str, error: str, finished_at: int) -> None:
        self._execute(
            """
            UPDATE crons
            SET status = 'error', updated_at = ?, completed_at = ?, last_error = ?
            WHERE id = ?
            """,
            (finished_at, finished_at, error, cron_id),
        )
