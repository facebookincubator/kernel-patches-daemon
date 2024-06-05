# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import io
import logging
import re
from abc import ABC, abstractmethod
from typing import Final, List, Optional, Sequence

import aiohttp
from github.WorkflowJob import WorkflowJob

from kernel_patches_daemon.status import gh_conclusion_to_status, Status

logger: logging.Logger = logging.getLogger(__name__)


class GithubFailedJobLog:
    def __init__(self, suite: str, arch: str, compiler: str, log: str, url: str):
        self._suite: str = suite
        self._arch: str = arch
        self._compiler: str = compiler
        self._log: str = log
        self._url: str = url

    @property
    def suite(self) -> str:
        return self._suite

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def compiler(self) -> str:
        return self._compiler

    @property
    def log(self) -> str:
        return self._log

    @property
    def url(self) -> str:
        return self._url

    @property
    def name(self) -> str:
        return f"{self._suite}-{self._arch}-{self._compiler}"


class GithubLogExtractor(ABC):
    @abstractmethod
    async def extract_failed_logs(
        self, jobs: Sequence[WorkflowJob]
    ) -> List[GithubFailedJobLog]:
        """
        Given a list of workflow jobs, `jobs`, filter out all the successful
        jobs. For the remaining failed jobs, pull out output logs. The logs
        will be minimally filtered. For maximal filtering, see
        generate_inline_email_text().
        """
        raise NotImplementedError

    @abstractmethod
    def generate_inline_email_text(self, logs: Sequence[GithubFailedJobLog]) -> str:
        """
        Given a list of failed job logs, return a (possibly multi-line) string
        suitable to be embedded in the body of a notification email. The text
        will try to be conservative -- high signal to email length is important.

        The returned string will include newline padding before and after the
        text (to make downstream formatting simpler).
        """
        raise NotImplementedError


class DefaultGithubLogExtractor(GithubLogExtractor):
    async def extract_failed_logs(
        self, jobs: Sequence[WorkflowJob]
    ) -> List[GithubFailedJobLog]:
        """Metadata parsing is tree-specific. So by default it's safer to do nothing."""
        return []

    def generate_inline_email_text(self, logs: Sequence[GithubFailedJobLog]) -> str:
        """Log parsing is also tree-specific."""
        return ""


class BpfGithubLogExtractor(GithubLogExtractor):
    TEST_PROGS_PREFIX: Final[str] = "test_progs"
    JOB_LOG_ERROR_START: Final[re.Pattern] = re.compile(".*##\\[group\\].*Error:.*")
    JOB_LOG_ERROR_END: Final[str] = "##[endgroup]"
    JOB_LOG_ERROR_MARKER: Final[str] = "##[error]"

    def __init__(self) -> None:
        # Needs to be initialized in async function
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return cached http session; creating if not already created"""
        if not self._session:
            # Read proxy from env var
            self._session = aiohttp.ClientSession(trust_env=True)

        return self._session

    async def _extract_job_log(self, job: WorkflowJob) -> Optional[GithubFailedJobLog]:
        status = gh_conclusion_to_status(job.conclusion)
        if status != Status.FAILURE:
            return None

        # NB: the job name is load bearing.
        #
        # Example names:
        #   x86_64-gcc / test (test_progs_no_alu32, false, 360) / test_progs_no_alu32 on x86_64 with gcc
        #   x86_64-llvm-17 / build / build for x86_64 with llvm-17
        job_name = [s.strip() for s in job.name.split("/")][-1]
        parts = job_name.split()
        if len(parts) != 5 or parts[1] not in ["on", "for"] or parts[3] != "with":
            logger.error(f"Invalid job name: '{job_name}', did workflow change?")
            return None

        suite = parts[0]
        arch = parts[2]
        compiler = parts[4]
        log = ""

        url = job.logs_url()
        session = await self._get_session()
        async with session.get(url) as resp:
            logger.info(f"Getting logs for {job.name} at {url}")
            if resp.ok:
                log = await resp.text()
            else:
                logger.warning(f"Failed to GET logs for {job.name}: HTTP {resp.status}")

        return GithubFailedJobLog(
            suite=suite,
            arch=arch,
            compiler=compiler,
            log=log,
            url=job.html_url,
        )

    async def extract_failed_logs(
        self, jobs: Sequence[WorkflowJob]
    ) -> List[GithubFailedJobLog]:
        tasks = [asyncio.create_task(self._extract_job_log(job)) for job in jobs]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    def _parse_out_test_progs_failure(self, log: str) -> str:
        # Avoid keeping a duplicate copy of a possibly large file in-memory
        log_file = io.StringIO(log)

        # Simple state machine track if we're looking at an error message
        in_error = False
        error_log = []

        # Example lines:
        # 2024-05-21T19:13:46.4638076Z ##[group][1;31mError:[0m #53 cgrp_local_storage
        # 2024-05-21T19:08:07.9400261Z ##[error]#53 cgrp_local_storage
        # 2024-05-21T19:08:07.9400806Z cgrp2_local_storage:PASS:join_cgroup /cgrp_local_storage 0 nsec
        # 2024-05-21T19:08:07.9401619Z ##[endgroup]
        for line in log_file:
            line = line.strip()

            if self.JOB_LOG_ERROR_START.match(line):
                in_error = True
                continue

            if self.JOB_LOG_ERROR_END in line:
                in_error = False
                continue

            if in_error:
                # Remove timestamp
                line = line[line.index(" ") + 1 :]

                # Remove ##[error] prefix on first line
                if line.startswith(self.JOB_LOG_ERROR_MARKER):
                    line = line[len(self.JOB_LOG_ERROR_MARKER) :]
                    if not line:
                        continue

                error_log.append(line)

        return "\n".join(error_log)

    def generate_inline_email_text(self, logs: Sequence[GithubFailedJobLog]) -> str:
        """
        Given a list of failed job logs, return a (possibly multi-line) string
        suitable to be embedded in the body of a notification email. The text
        will try to be conservative -- high signal to email length is important.

        The returned string will include newline padding before and after the
        text (to make downstream formatting simpler).
        """
        if not logs:
            return ""

        # Render header with links to failed jobs
        text = "\nFailed jobs:\n"
        for log in logs:
            text += f"{log.name}: {log.url}\n"

        # Render first test_progs failure
        for log in logs:
            if not log.suite.startswith(self.TEST_PROGS_PREFIX):
                continue

            error = self._parse_out_test_progs_failure(log.log)
            if not error:
                continue

            text += f"\nFirst test_progs failure ({log.name}):\n"
            text += f"{error}\n"
            break

        text += "\n"
        return text
