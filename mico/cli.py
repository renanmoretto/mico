import asyncio
from contextlib import suppress
from pathlib import Path

import typer

from .agent import Mico
from . import crons, storage

app = typer.Typer(add_completion=False, no_args_is_help=False)


async def _run_single_instance(
    db_path: str = '.db/mico.db',
) -> None:
    storage.init_storage(db_path=db_path)
    mico = Mico()
    cron_task = asyncio.create_task(crons.run_worker(mico=mico, poll_seconds=2.0, once=False))

    typer.echo(typer.style('Mico CLI', fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style(f'database: {Path(db_path).resolve()}', fg=typer.colors.BRIGHT_BLACK))
    typer.echo(typer.style('cron worker: enabled (single instance)', fg=typer.colors.BRIGHT_BLACK))
    typer.echo(typer.style('Type /exit to quit.', fg=typer.colors.BRIGHT_BLACK))

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(
                    typer.prompt,
                    typer.style('you', fg=typer.colors.GREEN, bold=True),
                )
            except (EOFError, KeyboardInterrupt):
                typer.echo('\nbye')
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input in {'/exit', '/quit'}:
                typer.echo('bye')
                break

            await mico.run(user_input=user_input)
    finally:
        cron_task.cancel()
        with suppress(asyncio.CancelledError):
            await cron_task


@app.callback(invoke_without_command=True)
def main(
    db_path: str = '.db/mico.db',
) -> None:
    asyncio.run(_run_single_instance(db_path=db_path))
