# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

import requests

from freezegun import freeze_time
from git.exc import GitCommandError
from kernel_patches_daemon.branch_worker import (
    BRANCH_TTL,
    BranchWorker,
    create_color_labels,
    has_same_base_different_remote,
    HEAD_BASE_SEPARATOR,
    PullRequest,
    temporary_patch_file,
    UPSTREAM_REMOTE_NAME,
)
from kernel_patches_daemon.patchwork import Series, Subject
from munch import Munch, munchify

from tests.common.patchwork_mock import (
    DEFAULT_TEST_RESPONSES,
    get_default_pw_client,
    pw_response_generator,
)


TEST_REPO = "repo"
TEST_REPO_URL = f"https://user:pass@127.0.0.1:0/org/{TEST_REPO}"
TEST_REPO_BRANCH = "test_branch"
TEST_REPO_PR_BASE_BRANCH = "test_branch_base"
TEST_UPSTREAM_REPO_URL = "https://127.0.0.2:0/upstream_org/upstream_repo"
TEST_UPSTREAM_BRANCH = "test_upstream_branch"
TEST_CI_REPO = "ci-repo"
TEST_CI_REPO_URL = f"https://user:pass@127.0.0.1:0/ci-org/{TEST_CI_REPO}"
TEST_CI_BRANCH = "test_ci_branch"
TEST_BASE_DIRECTORY = "/repos"
TEST_BRANCH = "test-branch"
TEST_CONFIG = {
    "version": 2,
    "project": "test",
    "pw_url": "pw",
    "pw_search_patterns": "pw-search-pattern",
    "pw_lookback": 5,
    "branches": {
        TEST_BRANCH: {
            "repo": TEST_REPO,
            "github_oauth_token": "test-oauth-token",
            "upstream": TEST_UPSTREAM_REPO_URL,
            "ci_repo": TEST_CI_REPO,
            "ci_branch": TEST_CI_BRANCH,
        }
    },
    "tag_to_branch_mapping": [],
}
TEST_LABELS_CFG = {
    "changes-requested": "2a76af",
    "merge-conflict": "e85506",
    "RFC": "f2e318",
    "new": "c2e0c6",
}
SERIES_DATA = {
    "id": 0,
    "name": "foo",
    "date": "2010-07-20T01:00:00",
    "patches": [],
    "cover_letter": None,
    "version": 0,
    "url": "https://example.com",
    "web_url": "https://example.com",
    "mbox": "https://example.com",
}


class BranchWorkerMock(BranchWorker):
    def __init__(self, *args, **kwargs) -> None:
        presets = {
            "patchwork": MagicMock(),
            "labels_cfg": TEST_LABELS_CFG,
            "repo_branch": TEST_REPO_BRANCH,
            "repo_url": TEST_REPO_URL,
            "github_oauth_token": "random_gh_oauth_token",
            "upstream_url": TEST_UPSTREAM_REPO_URL,
            "upstream_branch": TEST_UPSTREAM_BRANCH,
            "ci_repo_url": TEST_CI_REPO_URL,
            "ci_branch": TEST_CI_BRANCH,
            "base_directory": TEST_BASE_DIRECTORY,
        }
        presets.update(kwargs)

        super().__init__(*args, **presets)


def get_default_bw_client() -> BranchWorkerMock:
    return BranchWorkerMock()


class TestBranchWorker(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("kernel_patches_daemon.github_connector.Github")
        self._gh_mock = patcher.start()
        self.addCleanup(patcher.stop)
        # avoid local git commands
        patcher = patch("kernel_patches_daemon.branch_worker.git.Repo")
        self._git_repo_mock = patcher.start()
        self.addCleanup(patcher.stop)

        self._bw = BranchWorkerMock()

        # Patchwork client
        self._pw = get_default_pw_client()
        self._pw_post_patcher = patch.object(requests.Session, "post").start()
        self._pw_get_patcher = patch.object(requests.Session, "get").start()

    def test_fetch_repo_branch(self) -> None:
        """
        Fetch master fetches upstream repo, the CI repo, and check out the right branch.
        """
        with patch.object(BranchWorker, "fetch_repo") as fr:
            git_mock = unittest.mock.Mock()
            fr.return_value = git_mock
            self._bw.fetch_repo_branch()
            self.assertEqual(fr.call_count, 2)
            # We pull the repo with the right branch.
            self.assertEqual(fr.mock_calls[0].args[1], TEST_REPO_URL)
            self.assertEqual(fr.mock_calls[0].args[2], TEST_REPO_BRANCH)
            # Same for the CI repo.
            self.assertEqual(fr.mock_calls[1].args[1], TEST_CI_REPO_URL)
            self.assertEqual(fr.mock_calls[1].args[2], TEST_CI_BRANCH)
            # We check out the right branch.
            self.assertEqual(git_mock.mock_calls[0].args[0], f"origin/{TEST_CI_BRANCH}")

    def test_get_pulls(self) -> None:
        """
        When getting PR from GH some bookkeeping is done like checking relevancy of PR and state.
        """

        @dataclass
        class PR:
            state: str = "open"
            relevant: bool = True
            # Generate a random title... abuse Mock for this
            title: MagicMock = field(default_factory=MagicMock)

        @dataclass
        class TestCase:
            name: str
            prs: List[PR]
            added_pr_delta: int = 0
            relevant_prs_delta: int = 0

        test_cases = [
            TestCase(
                name="All open PRs, all relevant PRs",
                prs=[PR(), PR(), PR()],
                added_pr_delta=0,
                relevant_prs_delta=0,
            ),
            # Not supposed to happen, but we currently filter out non-opened PR regardless of the query to GH already doing that.
            TestCase(
                name="One non-opened, all relevant PRs",
                prs=[PR(), PR(state="invalid"), PR()],
                added_pr_delta=1,
                relevant_prs_delta=0,
            ),
            TestCase(
                name="All open PRs, one non-relevant PR",
                prs=[PR(), PR(relevant=False), PR()],
                added_pr_delta=0,
                relevant_prs_delta=1,
            ),
            # Not supposed to happen, but we currently filter out non-opened PR regardless of the query to GH already doing that.
            TestCase(
                name="One non-opened and non-relevant PR",
                prs=[PR(), PR(relevant=False, state="invalid"), PR()],
                added_pr_delta=1,
                relevant_prs_delta=1,
            ),
            # Not supposed to happen, but we currently filter out non-opened PR regardless of the query to GH already doing that.
            TestCase(
                name="One non-opened PR, one non-relevant PR",
                prs=[PR(), PR(relevant=False), PR(state="invalid"), PR()],
                added_pr_delta=1,
                relevant_prs_delta=1,
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                self._bw.repo.get_pulls.reset_mock()
                self._bw.repo.get_pulls.return_value = case.prs
                with patch.object(BranchWorker, "add_pr") as ap, patch.object(
                    BranchWorker, "_is_relevant_pr"
                ) as rp:
                    # set is_relevant return values
                    rp.side_effect = [pr.relevant for pr in case.prs]

                    self._bw.get_pulls()
                    # We get pull requests from upstream repo through Github API, only looking for opened PR against our branch of interest.
                    self._bw.repo.get_pulls.assert_called_once_with(
                        state="open", base=TEST_REPO_PR_BASE_BRANCH
                    )
                    # For each PR, we check that they are relevant
                    self.assertEqual(len(case.prs), rp.call_count)
                    # and only store the relevant ones
                    self.assertEqual(
                        len(case.prs) - len(self._bw.prs), case.relevant_prs_delta
                    )
                    # We only call add_pr for open PRs
                    self.assertEqual(len(case.prs) - ap.call_count, case.added_pr_delta)

    def test_do_sync_create_remote(self) -> None:
        """
        When syncing, if the remote does not exist in repo_local, create it.
        """

        # A fake remote
        @dataclass
        class R:
            name: str

        with patch.object(self._bw, "repo_local") as lr:
            lr.remotes = [
                R("a"),
                R("b"),
                R("c"),
            ]
            self._bw.do_sync()
            lr.create_remote.assert_called_once_with(
                UPSTREAM_REMOTE_NAME, TEST_UPSTREAM_REPO_URL
            )

            lr.delete_remote.assert_not_called()
            # After having created the remote, we access it.
            lr.remote.assert_called_once_with(UPSTREAM_REMOTE_NAME)

    def test_do_sync_recreate_remote_when_changed(self) -> None:
        """
        When syncing, if the remote exist in repo_local but is not tracking the
        right URL, delete and recreate it.
        """

        # A fake remote
        @dataclass
        class R:
            name: str

        with patch.object(self._bw, "repo_local") as lr:
            lr.remote.return_value = MagicMock(urls=["1", "2", "3"])
            lr.remotes = [
                R("a"),
                R("b"),
                R(UPSTREAM_REMOTE_NAME),
                R("c"),
            ]

            self._bw.do_sync()
            lr.create_remote.assert_called_once_with(
                UPSTREAM_REMOTE_NAME, TEST_UPSTREAM_REPO_URL
            )

            lr.delete_remote.assert_called_once_with(UPSTREAM_REMOTE_NAME)
            # We access the remote twice.
            self.assertEqual(2, lr.remote.call_count)
            lr.remote.assert_called_with(UPSTREAM_REMOTE_NAME)

    def test_do_sync_no_remote_modification(self) -> None:
        """
        If the remote is set correctly, we don't modify it.
        """
        # A fake remote
        @dataclass
        class R:
            name: str

        with patch.object(self._bw, "repo_local") as lr:
            lr.remote.return_value = MagicMock(urls=[self._bw.upstream_url])
            lr.remotes = [
                R("a"),
                R(UPSTREAM_REMOTE_NAME),
                R("b"),
            ]

            self._bw.do_sync()
            lr.create_remote.assert_not_called()
            lr.delete_remote.assert_not_called()
            # We access the remote twice.
            self.assertEqual(2, lr.remote.call_count)
            lr.remote.assert_called_with(UPSTREAM_REMOTE_NAME)

    def test_do_sync_reset_repo(self) -> None:
        with patch.object(self._bw, "repo_local") as lr, patch(
            "kernel_patches_daemon.branch_worker._reset_repo"
        ) as rr:
            # Create a mock suitable to mock a git.RemoteReference
            remote_ref = "a/b/c/d"
            m = MagicMock()
            m.__str__.return_value = remote_ref
            lr.remote.return_value.refs = MagicMock(**{TEST_UPSTREAM_BRANCH: m})
            self._bw.do_sync()

            # the repo is reset
            rr.assert_called_once()
            # and pushed from upstream_url to downstream
            lr.git.push.assert_called_once_with(
                "--force", "origin", f"{remote_ref}:refs/heads/{TEST_REPO_BRANCH}"
            )

    def test_relevant_pr(self) -> None:
        """
        Test to validate the combination of what make a PR relevant/irrelevant.
        """

        user_login = "foo"

        # Set our user login name
        self._bw.user_or_org = user_login

        def make_munch(
            head_user=user_login,
            base_user=user_login,
            base_ref=TEST_REPO_PR_BASE_BRANCH,
            state="open",
        ):
            """Helper to make a Munch that can be consumed as a PR (e.g accessing nested attributes)"""
            return munchify(
                {
                    "head": {"user": {"login": head_user}},
                    "base": {"user": {"login": base_user}, "ref": base_ref},
                    "state": state,
                }
            )

        @dataclass
        class TestCase:
            name: str
            pr: PullRequest
            relevant: bool

        test_cases = [
            TestCase(
                name="Relevant PR",
                pr=make_munch(),
                relevant=True,
            ),
            TestCase(
                name="Wrong head user",
                pr=make_munch(head_user="bar"),
                relevant=False,
            ),
            TestCase(
                name="Wrong base user",
                pr=make_munch(base_user="bar"),
                relevant=False,
            ),
            TestCase(
                name="Wrong head and base user",
                pr=make_munch(head_user="bar", base_user="bar"),
                relevant=False,
            ),
            TestCase(
                name="Wrong base ref",
                pr=make_munch(base_ref="some other base ref"),
                relevant=False,
            ),
            TestCase(
                name="Wrong state",
                pr=make_munch(state="some other state"),
                relevant=False,
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                self.assertEqual(self._bw._is_relevant_pr(case.pr), case.relevant)

    def test_fetch_repo_path_doesnt_exist_full_sync(self) -> None:
        """When the repo does not exist yet, a full sync is performed."""
        fetch_params = ["somepath", "giturl", "branch"]
        with patch.object(self._bw, "full_sync") as fr, patch(
            "kernel_patches_daemon.branch_worker.os.path.exists"
        ) as exists:
            # path does not exists
            exists.return_value = False
            self._bw.fetch_repo(*fetch_params)
            fr.assert_called_once_with(*fetch_params)

    def test_fetch_repo_path_exists_no_full_sync(self) -> None:
        """If the repo already exist, we don't perform a full sync."""
        fetch_params = ["somepath", "giturl", "branch"]
        with patch.object(self._bw, "full_sync") as fr, patch(
            "kernel_patches_daemon.branch_worker.os.path.exists"
        ) as exists:
            # path does exists
            exists.return_value = True
            self._bw.fetch_repo(*fetch_params)
            fr.assert_not_called()

    def test_fetch_repo_path_exists_git_exception(self) -> None:
        """When the repo exists but we hit a git command exception, we fallback on full sync."""
        fetch_params = ["somepath", "giturl", "branch"]
        with patch.object(self._bw, "full_sync") as fr, patch(
            "kernel_patches_daemon.branch_worker.os.path.exists"
        ) as exists:
            # path does exists
            exists.return_value = True
            self._git_repo_mock.init.return_value.git.fetch.side_effect = (
                # pyre-fixme[28]: Unexpected keyword argument `command`.
                GitCommandError(command="foo bar")
            )
            self._bw.fetch_repo(*fetch_params)
            fr.assert_called_once_with(*fetch_params)

    def test_expire_branches(self) -> None:
        """Only the branch that matches pattern and is expired should be deleted"""
        not_expired_time = datetime.fromtimestamp(3 * BRANCH_TTL)
        expired_time = datetime.fromtimestamp(BRANCH_TTL)

        @dataclass
        class PR:
            state: str = "closed"
            updated_at: datetime = not_expired_time

        @dataclass
        class TestCase:
            name: str
            branches: list
            all_prs: list = field(default_factory=list)
            # args and return PR for self.filter_closed_pr(branch)
            fcp_called_branches: list = field(default_factory=list)
            fcp_return_prs: list = field(default_factory=list)
            # args for self.delete_branch(branch)
            deleted_branches: list = field(default_factory=list)

        test_cases = [
            TestCase(
                name="A branch in all PR, even if supposedly expired, is not deleted.",
                branches=["test1", "test2"],
                all_prs=["test1", "test2"],
                # no filter_closed_pr call, no delete_branch call
            ),
            TestCase(
                name="A branch fetch pattern: expired should be deleted, not expired should not be deleted.",
                branches=[
                    "test1" + HEAD_BASE_SEPARATOR + self._bw.repo_branch,
                    "test2" + HEAD_BASE_SEPARATOR + self._bw.repo_branch,
                ],
                fcp_called_branches=[
                    "test1" + HEAD_BASE_SEPARATOR + self._bw.repo_branch,
                    "test2" + HEAD_BASE_SEPARATOR + self._bw.repo_branch,
                ],
                fcp_return_prs=[PR(updated_at=expired_time), PR()],
                deleted_branches=["test1" + HEAD_BASE_SEPARATOR + self._bw.repo_branch],
                # filter_closed_prs for both "test1=>repo_branch", "test2=>repo_branch"
                # only delete "test1=>repo_branch"
            ),
            TestCase(
                name="A branch that does not match the expected branch pattern, even if supposedly expired, is not deleted",
                branches=["test1", "test2"],
                # no filter_closed_pr call, no delete_branch call
            ),
            TestCase(
                name="A branch that belongs to self.repo_branch, even if supposedly expired, is not deleted",
                branches=[self._bw.repo_branch],
                # no filter_closed_pr call, no delete_branch call
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                self._bw.branches = case.branches
                self._bw.all_prs = case.all_prs
                with patch.object(self._bw, "filter_closed_pr") as fcp, patch.object(
                    self._bw, "delete_branch"
                ) as db, freeze_time(not_expired_time):
                    fcp.side_effect = case.fcp_return_prs
                    self._bw.expire_branches()
                    # check fcp and db are called with proper counts
                    self.assertEqual(len(case.fcp_called_branches), fcp.call_count)
                    self.assertEqual(len(case.deleted_branches), db.call_count)
                    # check args for each fcp called
                    self.assertEqual(
                        [x.args[0] for x in fcp.mock_calls], case.fcp_called_branches
                    )
                    # check args for each db called
                    self.assertEqual(
                        [x.args[0] for x in db.mock_calls], case.deleted_branches
                    )

    def test_filter_closed_pr(self) -> None:
        """Filter the most recent one closed PR per head from all closed PRs"""
        base_time = 10000
        base_datetime = datetime.fromtimestamp(base_time)

        def make_munch(
            head_ref="test",
            state="closed",
            updated_at=base_datetime,
            title="title",
        ):
            """Helper to make a Munch that can be consumed as a PR (e.g accessing nested attributes)"""
            return munchify(
                {
                    "head": {"ref": head_ref},
                    "state": state,
                    "updated_at": updated_at,
                    "title": title,
                }
            )

        @dataclass
        class TestCase:
            name: str
            closed_prs: List[Munch]
            branch: str
            return_pr: Munch

        test_cases = [
            TestCase(
                name="No PR should be returned",
                closed_prs=[
                    make_munch(head_ref="branch1"),
                    make_munch(head_ref="branch2"),
                ],
                branch="branch3",
                return_pr=make_munch(head_ref="None"),
            ),
            TestCase(
                name="PR with correct head should be returned",
                closed_prs=[
                    make_munch(head_ref="branch1", title="branch1"),
                    make_munch(head_ref="branch2", title="branch2"),
                ],
                branch="branch1",
                return_pr=make_munch(head_ref="branch1", title="branch1"),
            ),
            TestCase(
                name="The most recent one should be returned",
                closed_prs=[
                    make_munch(head_ref="branch1", title="branch1_old_pr"),
                    make_munch(
                        head_ref="branch1",
                        updated_at=datetime.fromtimestamp(base_time + 100),
                        title="branch1_recent_pr",
                    ),
                    make_munch(
                        head_ref="branch1",
                        updated_at=datetime.fromtimestamp(base_time + 50),
                        title="branch1_intermediary_pr",
                    ),
                ],
                branch="branch1",
                return_pr=make_munch(
                    head_ref="branch1",
                    updated_at=datetime.fromtimestamp(base_time + 100),
                    title="branch1_recent_pr",
                ),
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                with patch.object(self._bw, "closed_prs") as cp:
                    cp.return_value = case.closed_prs
                    return_pr = self._bw.filter_closed_pr(case.branch)
                    if not return_pr:
                        self.assertEqual("None", case.return_pr.head.ref)
                    else:
                        self.assertEqual(return_pr.title, case.return_pr.title)

    def test_delete_branches(self) -> None:
        """Delete a branch with correct calls and args"""
        branch_deleted = "branch"
        with patch.object(self._bw.repo, "get_git_ref") as ggr:
            self._bw.delete_branch(branch_deleted)
            ggr.assert_called_once_with(f"heads/{branch_deleted}")
            ggr.return_value.delete.assert_called_once()

    def test_guess_pr_return_from_active_pr_cache(self) -> None:
        # Whatever is in our self.prs's cache dictionary will be returned.
        series = Series(self._pw, SERIES_DATA)
        sentinel = random.random()
        self._bw.prs["foo"] = sentinel
        pr = self._bw._guess_pr(series)
        self.assertEqual(sentinel, pr)

    def test_guess_pr_return_from_secondary_cache_with_specified_branch(self) -> None:
        # After self.prs, we will look into self.all_prs
        # When calling _guess_pr with a branch name, we will look for it in
        # self.all_prs without trying to resolve the actual branch name based
        # on the series.
        mybranch = "mybranch"
        series = Series(self._pw, SERIES_DATA)
        sentinel = random.random()
        self._bw.all_prs[mybranch] = {}
        self._bw.all_prs[mybranch][TEST_REPO_BRANCH] = [sentinel]
        pr = self._bw._guess_pr(series, mybranch)
        self.assertEqual(sentinel, pr)

    def test_guess_pr_return_from_secondary_cache_without_specified_branch(
        self,
    ) -> None:
        # After self.prs, we will look into self.all_prs
        # When calling _guess_pr without a branch name, we will resolve it and
        # then look in self.all_prs.
        series = Series(self._pw, SERIES_DATA)
        mybranch = self._bw.subject_to_branch(Subject(series.subject, self._pw))

        sentinel = random.random()
        self._bw.all_prs[mybranch] = {}
        self._bw.all_prs[mybranch][TEST_REPO_BRANCH] = [sentinel]
        pr = self._bw._guess_pr(series, mybranch)
        self.assertEqual(sentinel, pr)

    def test_guess_pr_not_in_cache_no_specified_branch_no_remote_branch(self) -> None:
        """
        Handling of series which is not in our PR cache (self.prs, self.all_prs empty)
        and for which we do not have an active remote branch (self.branches)
        Repro for T1473514152
        """
        # Replace our BranchWorker PW instance by the mocked one.
        self._bw.patchwork = self._pw
        # Replace our BranchWorker repo instance by our gh_mock
        self._bw.repo = self._gh_mock

        series = Series(self._pw, {**SERIES_DATA, "name": "foo"})
        mybranch = self._bw.subject_to_branch(Subject(series.subject, self._pw))
        pr = self._bw._guess_pr(series, mybranch)

        # branch is not an active remote branch, we look up for existing PRs
        # and do not find any.
        self.assertTrue(self._gh_mock.method_calls)
        self.assertIsNone(pr)

    def test_guess_pr_not_in_cache_no_specified_branch_has_remote_branch_v1(
        self,
    ) -> None:
        """
        Handling of series which is not in our PR cache (self.prs, self.all_prs empty)
        and for which we do not have an active remote branch (self.branches).
        V1 series.
        Repro for T1473514152
        """
        # Replace our BranchWorker PW instance by the mocked one.
        self._bw.patchwork = self._pw
        # Replace our BranchWorker repo instance by our gh_mock
        self._bw.repo = self._gh_mock

        series = Series(self._pw, {**SERIES_DATA, "version": 1})
        mybranch = self._bw.subject_to_branch(Subject(series.subject, self._pw))
        self._bw.branches = "aaa"
        pr = self._bw._guess_pr(series, mybranch)

        # branch is an active remote branch, our series version is v1. We look up closed
        # PRs regardless but don't find any.
        self.assertTrue(self._gh_mock.method_calls)
        self.assertIsNone(pr)

    def test_guess_pr_not_in_cache_no_specified_branch_has_remote_branch_v2_first_series(
        self,
    ) -> None:
        """
        Handling of series which is not in our PR cache (self.prs, self.all_prs empty)
        and for which we do not have an active remote branch (self.branches)
        V2 series.
        Repro for T1473514152
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        # Replace our BranchWorker PW instance by the mocked one.
        self._bw.patchwork = self._pw
        # Replace our BranchWorker repo instance by our gh_mock
        self._bw.repo = self._gh_mock

        series = Series(self._pw, {**SERIES_DATA, "name": "code", "version": 2})
        mybranch = self._bw.subject_to_branch(Subject(series.subject, self._pw))
        self._bw.branches = [mybranch]
        pr = self._bw._guess_pr(series, mybranch)

        # branch is an active remote branch, we are on version 2, but we find only
        # one relevant series. We search for closed PRs regardless.
        self.assertTrue(self._gh_mock.method_calls)
        self.assertIsNone(pr)

    def test_guess_pr_not_in_cache_no_specified_branch_is_remote_branch_v2_multiple_series_noclosed_pr(
        self,
    ) -> None:
        """
        Handling of series which is not in our PR cache (self.prs, self.all_prs empty)
        and for which we do not have an active remote branch (self.branches).
        We look for a closed PR but don't find any.
        Repro for T1473514152
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        # Replace our BranchWorker PW instance by the mocked one.
        self._bw.patchwork = self._pw
        # Replace our BranchWorker repo instance by our gh_mock
        self._bw.repo = self._gh_mock
        series = Series(self._pw, {**SERIES_DATA, "name": "[v2] barv2", "version": 2})

        # DEFAULT_TEST_RESPONSES will return series 6 and 9, 6 being the first one
        mybranch = f"series/6=>{TEST_REPO_BRANCH}"
        self._bw.branches = [mybranch]

        # Calling without specifying `branch` so we force looking up series in
        # pw_tests.DEFAULT_TEST_RESPONSES
        pr = self._bw._guess_pr(series)

        # branch is an active remote branch, our series is on v2, we have
        # multiple relevant series so we lookup for closed PR but don't find any.
        self.assertTrue(self._gh_mock.method_calls)
        self.assertIsNone(pr)

    def test_guess_pr_not_in_cache_no_specified_branch_is_remote_branch_v2_multiple_series_with_closed_pr(
        self,
    ) -> None:
        """
        Handling of series which is not in our PR cache (self.prs, self.all_prs empty)
        and for which we do not have an active remote branch (self.branches)
        We look for a closed PR and find one.
        Repro for T1473514152
        """

        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        # Replace our BranchWorker PW instance by the mocked one.
        self._bw.patchwork = self._pw
        # Replace our BranchWorker repo instance by our gh_mock
        self._bw.repo = self._gh_mock
        # DEFAULT_TEST_RESPONSES will return series 6 and 9, 6 being the first one
        mybranch = f"series/6=>{TEST_REPO_BRANCH}"
        mymunch = munchify(
            {
                "head": {"ref": mybranch},
                "state": "closed",
                "updated_at": "2010-07-19T00:00:00",
                "title": "title",
            }
        )
        self._gh_mock.get_pulls.return_value = [
            mymunch,
        ]

        series = Series(self._pw, {**SERIES_DATA, "name": "[v2] barv2", "version": 2})

        self._bw.branches = [mybranch]

        # Calling without specifying `branch` so we force looking up series in
        # pw_tests.DEFAULT_TEST_RESPONSES
        pr = self._bw._guess_pr(series)

        # branch is an active remote branch, our series is on v2, we have
        # multiple relevant series so we lookup for closed PR and find one.
        self.assertTrue(self._gh_mock.method_calls)
        self.assertIsNotNone(pr)
        self.assertEqual(pr, mymunch)


class TestSupportFunctions(unittest.TestCase):
    def test_temporary_patch_file(self) -> None:
        content = b"test content"
        with temporary_patch_file(content) as tmp_file:
            self.assertEqual(content, tmp_file.read())

    def test_create_color_labels(self) -> None:
        with self.subTest("new"):
            repo_mock = MagicMock()
            create_color_labels({"label": "00000"}, repo_mock)

            repo_mock.get_labels.return_value = []
            repo_mock.create_label.assert_called_once_with(name="label", color="00000")

        with self.subTest("existing_update"):
            repo_label_mock = MagicMock()
            repo_label_mock.name = "label"
            repo_label_mock.color = "00001"

            repo_mock = MagicMock()
            repo_mock.get_labels.return_value = [repo_label_mock]

            create_color_labels({"label": "00000"}, repo_mock)

            repo_mock.create_label.assert_not_called()
            repo_label_mock.edit.assert_called_once_with(name="label", color="00000")

        with self.subTest("existing_skip"):
            repo_label_mock = MagicMock()
            repo_label_mock.name = "label"
            repo_label_mock.color = "00000"

            repo_mock = MagicMock()
            repo_mock.get_labels.return_value = [repo_label_mock]

            create_color_labels({"label": "00000"}, repo_mock)

            repo_mock.create_label.assert_not_called()
            repo_label_mock.edit.assert_not_called()

        with self.subTest("existing_case_mismatch"):
            repo_label_mock = MagicMock()
            repo_label_mock.name = "LabeL"
            repo_label_mock.color = "00001"

            repo_mock = MagicMock()
            repo_mock.get_labels.return_value = [repo_label_mock]

            create_color_labels({"laBel": "00000"}, repo_mock)

            repo_mock.create_label.assert_not_called()
            repo_label_mock.edit.assert_called_once_with(name="label", color="00000")

    def test_has_same_base_different_remote(self) -> None:
        with self.subTest("same_base_different_remote"):
            self.assertTrue(
                has_same_base_different_remote("base=>remote", "base=>other_remote")
            )

        with self.subTest("different_base_same_remote"):
            self.assertFalse(
                has_same_base_different_remote("base=>remote", "other_base=>remote")
            )

        with self.subTest("different_base_different_remote"):
            self.assertFalse(
                has_same_base_different_remote(
                    "base=>remote", "other_base=>other_remote"
                )
            )