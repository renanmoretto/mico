import re
import time


def now() -> int:
    return int(time.time())


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def truncate(text: str, limit: int = 12_000) -> str:
    return text[:limit] + '\n...[truncated]' if len(text) > limit else text


def slugify(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]+', '-', name.strip()).strip('-').lower() or 'agent'
