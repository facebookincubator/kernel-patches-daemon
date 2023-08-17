# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import time
from typing import Dict, Final, List, Optional, Sequence

from github import Auth
from github.PullRequest import PullRequest
from kernel_patches_daemon.branch_worker import (
    BranchWorker,
    get_base_branch_from_ref,
    has_same_base_different_remote,
    HEAD_BASE_SEPARATOR,
    NewPRWithNoChangeException,
)
from kernel_patches_daemon.config import BranchConfig, KPDConfig

from kernel_patches_daemon.patchwork import Patchwork, Series, Subject
from kernel_patches_daemon.stats import HistogramMetricTimer, Stats
from opentelemetry import metrics
from pyre_extensions import none_throws


logger: logging.Logger = logging.getLogger(__name__)
meter: metrics.Meter = metrics.get_meter("github_sync")

total_time: metrics.Histogram = meter.create_histogram(name="total_time")
patchwork_fetch_duration: metrics.Histogram = meter.create_histogram(
    name="patchwork_fetch.duration_ms"
)
bug_occurence: metrics.Counter = meter.create_counter(name="bug_occurence")
processed_prs: metrics.Counter = meter.create_counter(name="pull_requests.total")

git_clone_counter: metrics.Counter = meter.create_counter(name="clone")
git_clone_duration: metrics.Histogram = meter.create_histogram(name="clone.duration_ms")
git_fetch_counter: metrics.Counter = meter.create_counter(name="fetch")
git_fetch_duration: metrics.Histogram = meter.create_histogram(name="fetch.duration_ms")

github_ratelimit_remaining: metrics.Histogram = meter.create_histogram(
    name="ratelimit.remaining"
)

DEFAULT_HTTP_RETRIES: Final[int] = 3


def github_app_auth_from_branch_config(
    branch_config: BranchConfig,
) -> Optional[Auth.AppInstallationAuth]:
    if app_auth_config := branch_config.github_app_auth:
        return Auth.AppInstallationAuth(
            Auth.AppAuth(
                app_id=app_auth_config.app_id, private_key=app_auth_config.private_key
            ),
            installation_id=app_auth_config.installation_id,
        )
    else:
        return None


class GithubSync(Stats):
    def __init__(
        self,
        kpd_config: KPDConfig,
        labels_cfg: Dict[str, str],
        http_retries: int = DEFAULT_HTTP_RETRIES,
    ) -> None:
        self.pw = Patchwork(
            server=kpd_config.patchwork.base_url,
            search_patterns=kpd_config.patchwork.search_patterns,
            lookback_in_days=kpd_config.patchwork.lookback,
            auth_token=kpd_config.patchwork.token,
            http_retries=http_retries,
        )
        self.tag_to_branch_mapping = kpd_config.tag_to_branch_mapping
        self.workers: Dict[str, BranchWorker] = {
            branch: BranchWorker(
                patchwork=self.pw,
                labels_cfg=labels_cfg,
                repo_branch=branch,
                repo_url=branch_config.repo,
                upstream_url=branch_config.upstream_repo,
                upstream_branch=branch_config.upstream_branch,
                ci_repo_url=branch_config.ci_repo,
                ci_branch=branch_config.ci_branch,
                base_directory=kpd_config.base_directory,
                http_retries=http_retries,
                github_oauth_token=branch_config.github_oauth_token,
                app_auth=github_app_auth_from_branch_config(branch_config),
            )
            for branch, branch_config in kpd_config.branches.items()
        }

        # member variable initializations
        self.subjects: Sequence[Subject] = []
        super().__init__(
            {
                "full_cycle_duration",  # Duration of one sync cycle
                "mirror_duration",  # Duration of mirroring upstream
                "pw_fetch_duration",  # Duration of initial search in PW, exclusing time to map existing PRs to PW entities
                "patch_and_update_duration",  # Duration of git apply and git push
                "prs_total",  # All prs within the scope of work for this worker
                "empty_pr",  # Series that would result in an empty PR creation
                "all_known_subjects",  # All known subjects from PW and GH, including expired patches
            }
        )

    async def get_mapped_branches(self, series: Series) -> List[str]:
        for tag in self.tag_to_branch_mapping:
            if tag in await series.all_tags():
                mapped_branches = self.tag_to_branch_mapping[tag]
                logging.info(f"Tag '{tag}' mapped to branch order {mapped_branches}")
                return mapped_branches

        mapped_branches = self.tag_to_branch_mapping["__DEFAULT__"]
        logging.info(f"Mapped to default branch order: {mapped_branches}")
        return mapped_branches

    def close_existing_prs_with_same_base(
        self, workers: Sequence["BranchWorker"], pr: PullRequest
    ) -> None:
        """Close existing pull requests with the same base, but different remote branch

        For given pull request, find all other pull requests with
        the same base, but different remote branch and close them.
        """

        prs_to_close = [
            existing_pr
            for worker in workers
            for existing_pr in worker.prs.values()
            if has_same_base_different_remote(pr.head.ref, existing_pr.head.ref)
        ]
        # Remove matching PRs from other workers
        for pr_to_close in prs_to_close:
            logging.info(
                f"Closing existing pull request #{pr_to_close.number}, "
                f"replaced with #{pr.number}"
            )
            pr_to_close.edit(state="close")

            for worker in workers:
                if pr_to_close.title in worker.prs:
                    del worker.prs[pr_to_close.title]

    async def sync_patches(self) -> None:
        """
        One subject = one branch
        creates branches when necessary
        apply patches where it's necessary
        delete branches where it's necessary
        version of series applies in the same branch
        as separate commit
        """

        # sync mirror and fetch current states of PRs
        loop = asyncio.get_event_loop()

        self.drop_counters()
        sync_start = time.time()

        for branch, worker in self.workers.items():
            logging.info(f"Refreshing repo info for {branch}.")
            await loop.run_in_executor(None, worker.fetch_repo_branch)
            await loop.run_in_executor(None, worker.get_pulls)
            await loop.run_in_executor(None, worker.do_sync)
            worker._closed_prs = None
            worker.branches = [x.name for x in worker.repo.get_branches()]

        mirror_done = time.time()

        with HistogramMetricTimer(patchwork_fetch_duration):
            for branch, worker in self.workers.items():
                await loop.run_in_executor(
                    None, worker.update_e2e_test_branch_and_update_pr, branch
                )

            # fetch recent subjects
            self.subjects = await self.pw.get_relevant_subjects()

        pw_done = time.time()

        # 1. Get Subject's latest series
        # 2. Get series tags
        # 3. Map tags to a branches
        # 4. Start from first branch, try to apply and generate PR,
        #    if fails continue to next branch, if no more branches, generate a merge-conflict PR
        for subject in self.subjects:
            series = none_throws(await subject.latest_series)
            logging.info(
                f"Processing {series.id}: {subject.subject} (tags: {await series.all_tags()})"
            )

            mapped_branches = await self.get_mapped_branches(series)
            # series to apply - last known series
            last_branch = mapped_branches[-1]
            for branch in mapped_branches:
                worker = self.workers[branch]
                # PR branch name == sid of the first known series
                pr_branch_name = await worker.subject_to_branch(subject)
                apply_mbox = await worker.try_apply_mailbox_series(
                    pr_branch_name, series
                )
                if not apply_mbox[0]:
                    msg = f"Failed to apply series to {branch}, "
                    if branch != last_branch:
                        logging.info(msg + "moving to next.")
                        continue
                    else:
                        logging.info(msg + "no more next, staying.")
                logging.info(f"Choosing branch {branch} to create/update PR.")
                try:
                    self.increment_counter("all_known_subjects")
                    pr = await worker.checkout_and_patch(pr_branch_name, series)
                except NewPRWithNoChangeException:
                    self.increment_counter("empty_pr")
                    logger.exception("Could not create PR with no changes")

                    continue
                assert pr is not None
                logging.info(
                    f"Created/updated PR {pr.number}({pr.head.ref}): {pr.url} for series {series.id}"
                )
                await worker.sync_checks(pr, series)
                # Close out other PRs if exists
                self.close_existing_prs_with_same_base(list(self.workers.values()), pr)

                break

        # sync old subjects
        subject_names = {x.subject for x in self.subjects}
        for worker in self.workers.values():
            for subject_name, pr in worker.prs.items():
                if subject_name in subject_names:
                    continue

                if worker._is_relevant_pr(pr):
                    # ignore unknown format branch/PRs.
                    if "/" not in pr.head.ref or HEAD_BASE_SEPARATOR not in pr.head.ref:
                        continue

                    series_id = int(get_base_branch_from_ref(pr.head.ref.split("/")[1]))
                    series = await self.pw.get_series_by_id(series_id)
                    subject = self.pw.get_subject_by_series(series)
                    if subject_name != subject.subject:
                        logger.warning(
                            f"Renaming PR {pr.number} from {subject_name} to {subject.subject} according to {series.id}"
                        )
                        pr.edit(title=subject.subject)
                    branch_name = f"{await subject.branch}{HEAD_BASE_SEPARATOR}{worker.repo_branch}"
                    latest_series = await subject.latest_series or series
                    self.increment_counter("all_known_subjects")
                    await worker.checkout_and_patch(branch_name, latest_series)
                    await worker.sync_checks(pr, latest_series)

            await loop.run_in_executor(None, worker.expire_branches)
            await loop.run_in_executor(None, worker.expire_user_prs)

            rate_limit = worker.git.get_rate_limit()
            github_ratelimit_remaining.record(
                rate_limit.core.remaining,
                {"user": worker.github_account_name},
            )

        patches_done = time.time()
        self.set_counter("full_cycle_duration", patches_done - sync_start)
        total_time.record(patches_done - sync_start)
        self.set_counter("mirror_duration", mirror_done - sync_start)
        self.set_counter("pw_fetch_duration", pw_done - mirror_done)
        self.set_counter("patch_and_update_duration", patches_done - pw_done)
        for worker in self.workers.values():
            for pr in worker.prs.values():
                if worker._is_relevant_pr(pr):
                    self.increment_counter("prs_total")
                    processed_prs.add(1)
