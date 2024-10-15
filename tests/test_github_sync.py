# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from kernel_patches_daemon.branch_worker import NewPRWithNoChangeException
from kernel_patches_daemon.config import KPDConfig
from kernel_patches_daemon.github_sync import GithubSync

TEST_BRANCH = "test-branch"
TEST_CONFIG: dict[str, Any] = {
    "version": 3,
    "patchwork": {
        "project": "test",
        "server": "pw",
        "search_patterns": ["pw-search-pattern"],
        "lookback": 5,
    },
    "branches": {
        TEST_BRANCH: {
            "repo": "repo",
            "github_oauth_token": "test-oauth-token",
            "upstream": "https://127.0.0.2:0/upstream_org/upstream_repo",
            "ci_repo": "ci-repo",
            "ci_branch": "test_ci_branch",
        }
    },
    "tag_to_branch_mapping": {},
    "base_directory": "/tmp",
}


class GithubSyncMock(GithubSync):
    def __init__(
        self, kpd_config: KPDConfig | None = None, *args: Any, **kwargs: Any
    ) -> None:
        if kpd_config is None:
            kpd_config = KPDConfig.from_json(TEST_CONFIG)

        presets = {
            "kpd_config": kpd_config,
            "labels_cfg": {},
            **kwargs,
        }

        super().__init__(*args, **presets)


class TestGihubSync(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch("kernel_patches_daemon.github_connector.Github")
        self._gh_mock = patcher.start()
        self.addCleanup(patcher.stop)
        # avoid local git commands
        patcher = patch("kernel_patches_daemon.branch_worker.git.Repo")
        self._git_repo_mock = patcher.start()
        self.addCleanup(patcher.stop)

        self._gh = GithubSyncMock()

    def test_init_with_base_directory(self) -> None:
        @dataclass
        class TestCase:
            name: str
            prefix: str
            base_dir: str | None = None

        test_cases = [
            TestCase(
                name="base_directory not set in config",
                base_dir=None,
                prefix="/tmp/",
            ),
            TestCase(
                name="base_directory set in config",
                base_dir="/foo/bar/code",
                prefix="/foo/bar/code/",
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                config = copy.copy(TEST_CONFIG)
                if case.base_dir is not None:
                    config["base_directory"] = case.base_dir
                kpd_config = KPDConfig.from_json(config)
                gh = GithubSyncMock(kpd_config=kpd_config)
                self.assertEqual(
                    True,
                    gh.workers[TEST_BRANCH].repo_dir.startswith(case.prefix),
                )
                self.assertEqual(
                    True,
                    gh.workers[TEST_BRANCH].ci_repo_dir.startswith(case.prefix),
                )

    def test_close_existing_prs_with_same_base(self) -> None:
        matching_pr_mock = MagicMock()
        matching_pr_mock.title = "matching"
        matching_pr_mock.head.ref = "test_branch=>remote_branch"

        irrelevant_pr_mock = MagicMock()
        irrelevant_pr_mock.title = "irrelevant"
        irrelevant_pr_mock.head.ref = "other_branch=>other_remote"

        branch_worker_mock = MagicMock()
        branch_worker_mock.prs = {
            matching_pr_mock.title: matching_pr_mock,
            irrelevant_pr_mock.title: irrelevant_pr_mock,
        }

        input_pr_mock = MagicMock()
        input_pr_mock.head.ref = "test_branch=>other_remote_branch"

        workers = [copy.copy(branch_worker_mock) for _ in range(2)]
        self._gh.close_existing_prs_with_same_base(workers, input_pr_mock)
        for worker in workers:
            self.assertEqual(len(worker.prs), 1)
            self.assertTrue("irrelevant" in worker.prs)
            self.assertTrue("matching" not in worker.prs)

        matching_pr_mock.edit.assert_called_with(state="close")

    async def test_checkout_and_patch_safe(self) -> None:
        pr_branch_name = "fake_pr_branch"
        series = MagicMock()
        pr = MagicMock()
        branch_worker_mock = MagicMock()
        branch_worker_mock.checkout_and_patch = AsyncMock()

        # PR generated
        branch_worker_mock.checkout_and_patch.return_value = pr
        self.assertEqual(
            await self._gh.checkout_and_patch_safe(
                branch_worker_mock, pr_branch_name, series
            ),
            pr,
        )

        # One patch in series failed to apply
        branch_worker_mock.checkout_and_patch.return_value = None
        self.assertIsNone(
            await self._gh.checkout_and_patch_safe(
                branch_worker_mock, pr_branch_name, series
            )
        )

        # Series generates no changes vs target branch, likely already merged
        branch_worker_mock.checkout_and_patch.side_effect = NewPRWithNoChangeException(
            pr_branch_name, "target"
        )
        self.assertIsNone(
            await self._gh.checkout_and_patch_safe(
                branch_worker_mock, pr_branch_name, series
            )
        )
