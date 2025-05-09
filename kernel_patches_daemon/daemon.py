# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import logging
import signal
import threading
from typing import Callable, Dict, Final, Optional

from kernel_patches_daemon.config import KPDConfig
from kernel_patches_daemon.github_sync import GithubSync
from opentelemetry import metrics
from opentelemetry.metrics import Counter
from pyre_extensions import none_throws

logger: logging.Logger = logging.getLogger(__name__)
meter: metrics.Meter = metrics.get_meter("worker")

success_counter: Counter = meter.create_counter(name="runs.success")
fail_counter: Counter = meter.create_counter(name="runs.failed")

DEFAULT_LOOP_DELAY: Final[int] = 120
DEFAULT_MAX_CONCURRENT_RESTARTS: Final[int] = 5


class KernelPatchesWorker:
    def __init__(
        self,
        kpd_config: KPDConfig,
        labels_cfg: Dict[str, str],
        metrics_logger: Optional[Callable] = None,
        loop_delay: int = DEFAULT_LOOP_DELAY,
        max_concurrent_restarts: int = DEFAULT_MAX_CONCURRENT_RESTARTS,
    ) -> None:
        self.project: str = kpd_config.patchwork.project
        self.github_sync_worker = GithubSync(
            kpd_config=kpd_config, labels_cfg=labels_cfg
        )
        self.max_concurrent_restarts: Final[int] = max_concurrent_restarts
        self.loop_delay: Final[int] = loop_delay
        self.metrics_logger = metrics_logger

    async def run_once(self) -> None:
        await self.github_sync_worker.sync_patches()
        logger.info("Submitting run metrics into metrics logger")
        if self.metrics_logger:
            self.metrics_logger(self.project, self.github_sync_worker.stats)

    async def run(self) -> None:
        concurrent_restarts = 0
        while True:
            try:
                await self.run_once()
                concurrent_restarts = 0
                success_counter.add(1)
            except Exception as e:
                fail_counter.add(1)
                concurrent_restarts += 1
                if self.max_concurrent_restarts < concurrent_restarts:
                    raise e

                logger.exception(
                    "Kernel Patches Daemon crashed, restarting... "
                    f"remaining attempts: [{concurrent_restarts}/{self.max_concurrent_restarts}]"
                )
            logger.info(f"Waiting for {self.loop_delay} seconds before next run...")
            await asyncio.sleep(self.loop_delay)


class KernelPatchesDaemon:
    def __init__(
        self,
        kpd_config: KPDConfig,
        labels_cfg: Dict[str, str],
        metrics_logger: Optional[Callable] = None,
        max_concurrent_restarts: int = 1,
    ) -> None:
        self._stopping_lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        self.worker = KernelPatchesWorker(
            kpd_config=kpd_config,
            labels_cfg=labels_cfg,
            metrics_logger=metrics_logger,
            max_concurrent_restarts=max_concurrent_restarts,
        )

    def stop(self) -> None:
        if not self._task:
            logger.info("Kernel Patches Daemon was never started")

        with self._stopping_lock:
            logger.info("Stopping Kernel Patches Daemon...")
            none_throws(self._task).cancel()
            logger.info("Kernel Patches Daemon stopped")

    async def start_async(self) -> None:
        logger.info("Starting Kernel Patches Daemon...")

        loop = asyncio.get_event_loop()

        loop.add_signal_handler(signal.SIGTERM, self.stop)
        loop.add_signal_handler(signal.SIGINT, self.stop)

        self._task = asyncio.create_task(self.worker.run())
        await asyncio.gather(self._task, return_exceptions=True)

    def start(self) -> None:
        asyncio.run(self.start_async())
