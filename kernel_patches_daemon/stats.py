# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import time
from collections.abc import Iterable

from opentelemetry import metrics
from opentelemetry.util.types import Attributes
from pyre_extensions import none_throws

STATS_KEY_BUG: str = "bug_occurence"
DEFAULT_STATS: set[str] = {STATS_KEY_BUG}


logger: logging.Logger = logging.getLogger(__name__)


class Timer:
    def __init__(self) -> None:
        self._start_time: int | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self._start_time = None

    def start(self) -> None:
        if self._start_time is not None:
            raise RuntimeError("Timer is already started")
        self._start_time = time.time_ns()

    def stop(self) -> int:
        elapsed_time = self.elapsed()
        self._start_time = None
        return elapsed_time

    def elapsed(self) -> int:
        if not self._start_time:
            raise RuntimeError("Timer is not running")

        return time.time_ns() - none_throws(self._start_time)

    def elapsed_ms(self) -> float:
        return self.elapsed() / 1_000_000

    def elapsed_sec(self) -> float:
        return self.elapsed_ms() / 1_000


class HistogramMetricTimer(Timer):
    def __init__(
        self, metric: metrics.Histogram, attributes: Attributes | None = None
    ) -> None:
        self.metric: metrics.Histogram = metric
        self.attributes = attributes
        super().__init__()

    def __exit__(self, *exc_info) -> None:
        self.metric.record(self.elapsed_ms(), attributes=self.attributes)
        super().__exit__(exc_info)


class Stats:
    def __init__(self, counters: Iterable[str]) -> None:
        self.counters: set[str] = set(counters)
        self.stats: dict = {}
        self.drop_counters()

    def drop_counters(self) -> None:
        for counter in DEFAULT_STATS | self.counters:
            self.stats[counter] = 0

    def increment_counter(self, key: str, increment: int = 1) -> None:
        try:
            self.stats[key] += increment
        except Exception:
            self.stats[STATS_KEY_BUG] += 1
            logger.error(f"Failed to add {increment} increment to '{key}' stat")

    def set_counter(self, key: str, value: int | float) -> None:
        try:
            self.stats[key] = value
        except Exception:
            self.stats[STATS_KEY_BUG] += 1
            logger.error(f"Failed to set {value} for '{key}' stat")
