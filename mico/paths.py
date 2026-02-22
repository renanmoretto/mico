from pathlib import Path

MICO_DIR = Path('.mico')


def db_path() -> str:
    return str(MICO_DIR / 'mico.db')


def agents_dir() -> str:
    return str(MICO_DIR / 'agents')
