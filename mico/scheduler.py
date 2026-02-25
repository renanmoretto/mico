from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from croniter import croniter

from . import storage
from .logging import logger

if TYPE_CHECKING:
    from .agent import Mico


class SchedulerWorker:
    def __init__(self, mico: Mico, poll_interval: float = 30.0):
        self._mico = mico
        self._poll_interval = poll_interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name='scheduler-poll')
        logger.info(f'SchedulerWorker started (poll_interval={self._poll_interval}s)')

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info('SchedulerWorker stopped')

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('SchedulerWorker tick failed')
            await asyncio.sleep(self._poll_interval)

    async def _tick(self) -> None:
        now = int(time.time())
        due_jobs = await storage.get_due_jobs(now)
        if not due_jobs:
            return
        logger.info(f'Scheduler: {len(due_jobs)} due job(s) found')
        for job in due_jobs:
            try:
                await self._execute_job(job)
            except Exception:
                logger.exception(f'Scheduler: failed to execute job {job.id}')

    async def _execute_job(self, job: storage.ScheduledJobRecord) -> None:
        logger.info(f'Scheduler: executing job {job.id} ({job.description}) for agent {job.agent_id[:8]}')
        now = int(time.time())

        agent_row = await storage.get_agent(job.agent_id)
        if agent_row is None or agent_row.status == 'deleted':
            logger.warning(f'Scheduler: agent {job.agent_id[:8]} not found for job {job.id}, deleting job')
            await storage.delete_scheduled_job(agent_id=job.agent_id, job_id=job.id)
            return

        try:
            await self._mico.run(
                agent_id=job.agent_id,
                system_input=job.instruction,
                metadata={'source': 'scheduler', 'job_id': job.id},
            )
        except Exception:
            logger.exception(f'Scheduler: Mico.run failed for job {job.id}')

        if job.job_type == 'once':
            await storage.update_job_after_run(job_id=job.id, next_run_at=None, status='completed', last_run_at=now)
            logger.info(f'Scheduler: one-time job {job.id} completed')
        else:
            # Recurring: compute next future run time from now
            dt_now = datetime.fromtimestamp(now, tz=timezone.utc)
            cron = croniter(job.cron_expr, dt_now)
            next_dt = cron.get_next(datetime)
            next_run_at = int(next_dt.timestamp())
            await storage.update_job_after_run(job_id=job.id, next_run_at=next_run_at, status='active', last_run_at=now)
            logger.info(f'Scheduler: recurring job {job.id} next run at {next_run_at}')
