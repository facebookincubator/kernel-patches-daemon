# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import copy
import email
import email.parser
import email.policy
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from subprocess import PIPE
from typing import Any, Dict, Final, Generator, IO, List, Optional, Sequence, Tuple

import dateutil.parser
import git
from github import Auth, GithubException
from github.Label import Label as GithubLabel
from github.PullRequest import PullRequest
from github.Repository import Repository
from github.WorkflowJob import WorkflowJob

from kernel_patches_daemon.config import EmailConfig
from kernel_patches_daemon.github_connector import GithubConnector
from kernel_patches_daemon.github_logs import GithubFailedJobLog, GithubLogExtractor
from kernel_patches_daemon.patchwork import Patchwork, Series, Subject
from kernel_patches_daemon.stats import HistogramMetricTimer
from kernel_patches_daemon.status import (
    gh_conclusion_to_status,
    process_statuses,
    Status,
)
from kernel_patches_daemon.utils import redact_url, remove_unsafe_chars
from opentelemetry import metrics
from pyre_extensions import none_throws


logger: logging.Logger = logging.getLogger(__name__)
meter: metrics.Meter = metrics.get_meter("branch_worker")

git_clone_counter: metrics.Counter = meter.create_counter(name="clone")
git_clone_duration: metrics.Histogram = meter.create_histogram(name="clone.duration_ms")
git_fetch_counter: metrics.Counter = meter.create_counter(name="fetch")
git_fetch_duration: metrics.Histogram = meter.create_histogram(name="fetch.duration_ms")
pr_summary_report: metrics.Counter = meter.create_counter(name="pr_summary_reports")
email_success_counter: metrics.Counter = meter.create_counter(name="email.success")
email_failure_counter: metrics.Counter = meter.create_counter(name="email.failure")
email_conflict_counter: metrics.Counter = meter.create_counter(name="email.conflict")
email_send_fail_counter: metrics.Counter = meter.create_counter(name="email.send_fail")
pr_created: metrics.Counter = meter.create_counter(name="pull_requests.created")
pr_updated: metrics.Counter = meter.create_counter(name="pull_requests.updated")
pr_closed: metrics.Counter = meter.create_counter(name="pull_requests.closed")
pr_merge_conflict: metrics.Counter = meter.create_counter(
    name="pull_requests.merge_conflict"
)
branch_deleted: metrics.Counter = meter.create_counter(name="branches.deleted")
errors: metrics.Counter = meter.create_counter(name="errors")

# Since under normal conditions the race is quite small, the last
# 100 commits should be sufficient. We could always increase the
# count but it would slow down normal operation more.
ALREADY_MERGED_LOOKBACK = 100
BRANCH_TTL = 172800  # 1 week
PULL_REQUEST_TTL = timedelta(days=7)
HEAD_BASE_SEPARATOR = "=>"
KNOWN_OK_COMMENT_EXCEPTIONS = {
    "Commenting is disabled on issues with more than 2500 comments"
}
CI_APP = 15368  # GithubApp(url="/apps/github-actions", id=15368)
CI_VMTEST_NAME = "VM_Test"

CI_DESCRIPTION = "vmtest"
MERGE_CONFLICT_LABEL = "merge-conflict"
UPSTREAM_REMOTE_NAME = "upstream"


EMAIL_TEMPLATE_BASE: Final[str] = """\
Dear patch submitter,

CI has tested the following submission:
Status:     {status}
Name:       {submission_name}
Patchwork:  {pw_series_url}
{body}

Please note: this email is coming from an unmonitored mailbox. If you have
questions or feedback, please reach out to the Meta Kernel CI team at
kernel-ci@meta.com.
"""

EMAIL_TEMPLATE_MERGE_CONFLICT_BODY: Final[str] = """\
PR:         {github_pr_url}

Please rebase your submission onto the most recent upstream change and resubmit
the patch to get it tested again.
"""

EMAIL_TEMPLATE_SUCCESS_BODY: Final[str] = """\
Matrix:     {github_actions_url}

No further action is necessary on your part.
"""

EMAIL_TEMPLATE_FAILURE_BODY: Final[str] = """\
Matrix:     {github_actions_url}

{inline_logs}\
"""


# Context used to format email notification body
EmailBodyContext = namedtuple(
    "EmailBodyContext", "status submission_name patchwork_url github_url inline_logs"
)


class StatusLabelSuffixes(Enum):
    PASS = "ci-pass"
    FAIL = "ci-fail"

    @classmethod
    def all(cls):
        return [c.value for c in cls]

    def to_label(self, version: int) -> str:
        return f"V{version}-{self.value}"


class NewPRWithNoChangeException(Exception):
    def __init__(self, base_branch, target_branch, *args):
        super().__init__(args)
        self.base_branch = base_branch
        self.target_branch = target_branch

    def __str__(self):
        return (
            f"No changes between {self.base_branch} and {self.target_branch}, "
            "cannot create PR if there is no change. Was this series/patches recently merged?"
        )


def get_ci_base(series: Series) -> Dict:
    """Retrieve the object (cover letter or patch) that we use as the base for
    sending emails in response to.
    """
    if series.cover_letter is not None:
        return series.cover_letter
    else:
        return series.patches[-1]


def get_github_actions_url(repo: Repository, pr: PullRequest, status: Status) -> str:
    """Find a URL representing a GitHub Actions run for the given pull
    request with the given status.
    """
    # If we can't find a matching run we just work with the pull request
    # URL.
    github_actions_url = pr.html_url
    # Attempt to retrieve the URL to the workflow summary page.
    for run in repo.get_workflow_runs(head_sha=pr.head.sha):
        # We use the page of the first run that has the conclusion we
        # report.
        if gh_conclusion_to_status(run.conclusion) == status:
            github_actions_url = run.html_url
            break

    return github_actions_url


async def get_ci_email_subject(series: Series) -> str:
    """Get the subject to use for a CI email pertaining the given series."""
    obj = get_ci_base(series)
    patch = await series.pw_client.get_blob(obj["mbox"])
    parser = email.parser.BytesParser(policy=email.policy.default)
    msg = parser.parsebytes(patch, headersonly=True)
    return f"Re: {msg.get('subject', series.name)}"


def build_email_body_context(
    repo: Repository, pr: PullRequest, status: Status, series: Series, inline_logs: str
) -> EmailBodyContext:
    """
    Generate a context to be used for formatting email notification body.

    This is used as an extra indirection to make testing easier.
    """
    return EmailBodyContext(
        status=status,
        submission_name=get_ci_base(series)["name"],
        patchwork_url=series.web_url + "&state=*",
        github_url=get_github_actions_url(repo, pr, status),
        inline_logs=inline_logs,
    )


def furnish_ci_email_body(ctx: EmailBodyContext) -> str:
    """Prepare the body of a BPF CI email according to the provided context"""
    if ctx.status == Status.SUCCESS:
        body = EMAIL_TEMPLATE_SUCCESS_BODY.format(github_actions_url=ctx.github_url)
    elif ctx.status == Status.FAILURE:
        body = EMAIL_TEMPLATE_FAILURE_BODY.format(
            inline_logs=ctx.inline_logs,
            github_actions_url=ctx.github_url,
        )
    else:
        assert ctx.status == Status.CONFLICT
        body = EMAIL_TEMPLATE_MERGE_CONFLICT_BODY.format(github_pr_url=ctx.github_url)

    return EMAIL_TEMPLATE_BASE.format(
        status=str(ctx.status.value).upper(),
        submission_name=ctx.submission_name,
        pw_series_url=ctx.patchwork_url,
        body=body,
    )


def bump_email_status_counters(status: Status):
    if status == Status.SUCCESS:
        email_success_counter.add(1)
    elif status == Status.FAILURE:
        email_failure_counter.add(1)
    else:
        email_conflict_counter.add(1)


def generate_msg_id(host: str) -> str:
    """Generate an email message ID based on the provided host."""
    checksum = hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()
    return f"{checksum}@{host}"


def email_in_submitter_allowlist(email: str, allowlist: Sequence[re.Pattern]) -> bool:
    """
    Checks if an email is in the submitter allowlist

    Note that there may be false positives when folks have regex syntax in
    their email address. But that is ok -- this is simply a rollout mechanism.
    We only need to roughly control the rollout.
    """
    return any(regex.fullmatch(email) for regex in allowlist)


def build_email(
    config: EmailConfig,
    series: Series,
    subject: str,
    msg_id: str,
    body: str,
    boundary: str = "",
) -> Tuple[List[str], str]:
    """
    Builds complete email (including headers) to be sent along with curl command
    necessary to send it.

    `boundary` is only used for unit tests to make the content-type boundary
    predictable.

    Returns (command, email).
    """
    # We work with curl as it's the only thing that supports SMTP via an HTTP
    # proxy, which is something we require for production environments.
    args = [
        "curl",
        "--silent",
        "--show-error",
        "--ssl-reqd",
        f"smtps://{config.smtp_host}",
        "--mail-from",
        config.smtp_from,
        "--user",
        f"{config.smtp_user}:{config.smtp_pass}",
        "--crlf",
        "--upload-file",
        "-",
    ]

    to_list = copy.copy(config.smtp_to)
    cc_list = copy.copy(config.smtp_cc)

    if config.ignore_allowlist or email_in_submitter_allowlist(
        series.submitter_email, config.submitter_allowlist
    ):
        to_list += [series.submitter_email]

    for to in to_list + cc_list:
        args += ["--mail-rcpt", to]

    if config.smtp_http_proxy is not None:
        args += ["--proxy", config.smtp_http_proxy]

    msg = MIMEMultipart()
    # Add some RFC 822 style message ID to allow for easier referencing of this
    # message. Note that it's not entirely correct for us to refer to a host
    # that is not entirely under our control, but we don't want to expose our
    # actual host name either. Collisions of a sha256 hash are assumed to be
    # unlikely in many contexts, so we do the same.
    msg["Message-Id"] = f"<{msg_id}>"
    msg["In-Reply-To"] = get_ci_base(series)["msgid"]
    msg["References"] = msg["In-Reply-To"]
    msg["Subject"] = subject
    msg["From"] = config.smtp_from
    if to_list:
        msg["To"] = ",".join(to_list)
    if cc_list:
        msg["Cc"] = ",".join(cc_list)
    if boundary:
        msg.set_boundary(boundary)
    msg.attach(MIMEText(body, "plain"))

    return (args, msg.as_string())


async def send_email(
    config: EmailConfig,
    series: Series,
    subject: str,
    body: str,
):
    """Send an email."""
    msg_id = generate_msg_id(config.smtp_host)
    curl_args, msg = build_email(config, series, subject, msg_id, body)

    proc = await asyncio.create_subprocess_exec(
        *curl_args, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    stdout, stderr = await proc.communicate(input=msg.encode())
    rc = await proc.wait()
    if rc != 0:
        logger.error(f"failed to send email: {stdout.decode()} {stderr.decode()}")
        email_send_fail_counter.add(1)


def _is_pr_flagged(pr: PullRequest) -> bool:
    for label in pr.get_labels():
        if MERGE_CONFLICT_LABEL == label.name:
            return True
    return False


def execute_command(cmd: str) -> None:
    logger.info(cmd)
    os.system(cmd)


def _uniq_tmp_folder(
    url: Optional[str], branch: Optional[str], base_directory: str
) -> str:
    # use same folder for multiple invocation to avoid cloning whole tree every time
    # but use different folder for different workers identified by url and branch name
    sha = hashlib.sha256()
    sha.update(f"{url}/{branch}".encode("utf-8"))
    # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <: [str,
    #  bytes]]]` but got `Optional[str]`.
    repo_name = remove_unsafe_chars(os.path.basename(url))
    return os.path.join(base_directory, f"pw_sync_{repo_name}_{sha.hexdigest()}")


@contextmanager
def temporary_patch_file(content: bytes) -> Generator[IO, None, None]:
    """
    Create a temporary file with given content and return IO object with a pointer set to the beginning to the file.
    """
    tmp_patch_file = tempfile.NamedTemporaryFile(prefix="kpd_tmp_patch", mode="w+b")
    try:
        tmp_patch_file.write(content)
        tmp_patch_file.seek(0)
        yield tmp_patch_file
    finally:
        tmp_patch_file.close()


def create_color_labels(labels_cfg: Dict[str, str], repo: Repository) -> None:
    repo_labels: Dict[str, GithubLabel] = {x.name.lower(): x for x in repo.get_labels()}
    labels_cfg: Dict[str, str] = {k.lower(): v for k, v in labels_cfg.items()}

    for label, color in labels_cfg.items():
        if repo_label := repo_labels.get(label):
            if repo_label.name != label or repo_label.color != color:
                repo_label.edit(name=label, color=color)
        else:
            repo.create_label(name=label, color=color)


def get_base_branch_from_ref(ref: str) -> str:
    return ref.split(HEAD_BASE_SEPARATOR)[0]


def has_same_base_different_remote(ref: str, other_ref: str) -> bool:
    if ref == other_ref:
        return False

    base = get_base_branch_from_ref(ref)
    other_base = get_base_branch_from_ref(other_ref)

    if base != other_base:
        return False

    return True


def _reset_repo(repo, branch: str) -> None:
    """
    Reset the repository into a known good state, with `branch` checked out.
    """
    try:
        repo.git.am("--abort")
    except git.exc.GitCommandError:
        pass

    repo.git.reset("--hard", branch)
    repo.git.checkout(branch)


async def _series_already_applied(repo: git.Repo, series: Series) -> bool:
    """
    Returns whether or not the given series has already been applied
    to the active branch on `repo`.

    We consider even a single applied patch in a series as the series being
    applied. This is b/c the applied patch is marked as "accepted" and thus the
    entire series is marked as closed. See patchwork.py::is_closed.

    Note we are only checking the commit summaries. We have to be
    conservative b/c the commit messages will have trailers attached
    upon merge. And we cannot check the commit diff b/c maintainers
    could have applied a fixup.
    """
    try:
        summaries = {
            commit.summary.lower()
            for commit in repo.iter_commits(max_count=ALREADY_MERGED_LOOKBACK)
        }
    except git.exc.GitCommandError:
        logger.exception("Failed to check series application status")
        return False

    return any(ps.lower() in summaries for ps in await series.patch_subjects())


def _is_branch_changed(
    repo: git.Repo, base_branch: str, old_branch: str, new_branch: str
) -> bool:
    """
    Returns whether or not the new_branch (in local repo) is different from
    the old_branch (remote repo).

    Note that care is taken to avoid detecting commit SHA differences. The
    SHA is unstable even if the contents are the same. So compare the contents
    we care about.
    """
    # Check if code changes are different
    if repo.git.diff(new_branch, old_branch):
        return True

    # Check if change metadata is different (number of commits and
    # commit messages)
    old_metadata = repo.git.log('--format="%B"', f"{base_branch}..{old_branch}")
    new_metadata = repo.git.log('--format="%B"', f"{base_branch}..{new_branch}")
    if old_metadata != new_metadata:
        return True

    return False


def _is_outdated_pr(pr: PullRequest) -> bool:
    """
    Check if a pull request is outdated, i.e., whether it has not seen an update recently.
    """
    commits = pr.get_commits()
    if commits.totalCount == 0:
        return False

    # We only look at the most recent commit, which should be the last one. If
    # a user modified an earlier one, the magic of Git ensures that any commits
    # that have it as a parent will also be changed.
    commit = commits[commits.totalCount - 1]
    last_modified = dateutil.parser.parse(commit.stats.last_modified)
    age = datetime.now(timezone.utc) - last_modified
    logger.info(f"Pull request {pr} has age {age}")
    return age > PULL_REQUEST_TTL


class BranchWorker(GithubConnector):
    @staticmethod
    def slugify_context(s: str):
        # According to patchwork rule:
        # https://github.com/getpatchwork/patchwork/blob/2aa4742ec88be4cd07f569805d22a35c08a08f40/releasenotes/notes/slugify-check-context-dc586f204b5058a7.yaml
        # the context need to be a slug, or a string consisting of only
        # ASCII letters, numbers, underscores or hyphens.
        # Lets replace all "." with "_"
        return s.replace(".", "_")

    def __init__(
        self,
        patchwork: Patchwork,
        labels_cfg: Dict[str, Any],
        repo_branch: str,
        repo_url: str,
        upstream_url: str,
        upstream_branch: str,
        base_directory: str,
        ci_repo_url: str,
        ci_branch: str,
        log_extractor: GithubLogExtractor,
        github_oauth_token: Optional[str] = None,
        app_auth: Optional[Auth.AppInstallationAuth] = None,
        email: Optional[EmailConfig] = None,
        http_retries: Optional[int] = None,
    ) -> None:
        super().__init__(
            repo_url=repo_url,
            github_oauth_token=github_oauth_token,
            app_auth=app_auth,
            http_retries=http_retries,
        )

        self.patchwork = patchwork
        self.email = email

        self.log_extractor = log_extractor
        self.ci_repo_url = ci_repo_url
        self.ci_repo_dir = _uniq_tmp_folder(ci_repo_url, ci_branch, base_directory)
        self.ci_branch = ci_branch
        # Set properly at a later time.

        self.repo_dir = _uniq_tmp_folder(repo_url, repo_branch, base_directory)
        self.repo_branch = repo_branch
        self.repo_pr_base_branch = repo_branch + "_base"
        self.repo_local = None

        self.upstream_url = upstream_url
        self.upstream_branch = upstream_branch
        # Most recently used upstream SHA-1. Used to prevent unnecessary pushes
        # if upstream did not change.
        self.upstream_sha = None

        create_color_labels(labels_cfg, self.repo)
        # member variables
        self.branches = {}
        self.prs: Dict[str, PullRequest] = {}
        self.all_prs = {}
        self._closed_prs = None

    def _create_new_pull_request(
        self, title: str, message: str, head: str, base: str
    ) -> PullRequest:
        logger.info(f"Creating new pull request '{title}': {head} => {base}")
        pr = self.repo.create_pull(title=title, body=message, head=head, base=base)
        self.prs[title] = pr
        self.add_pr(pr)
        return pr

    def _add_pull_request_comment(self, pr: PullRequest, message: str) -> None:
        try:
            pr.create_issue_comment(message)
        except GithubException as e:
            if not isinstance(e.data, Dict):
                raise e
            emsg = e.data.get("message")
            if emsg is not None and emsg in KNOWN_OK_COMMENT_EXCEPTIONS:
                logger.warning(
                    f"Expected exception while adding a comment to {pr}: {emsg}"
                )
            else:
                raise e

    def _update_e2e_pr(
        self, title: str, base_branch: str, branch: str, has_codechange: bool
    ) -> Optional[PullRequest]:
        """Check if there is open PR on e2e branch, reopen if necessary."""
        pr = None

        message = f"branch: {branch}\nbase: {self.repo_branch}\nversion: {self.upstream_sha}\n"

        if title in self.prs:
            pr = self.prs[title]

        if pr:
            if pr.state == "closed":
                pr = None
            elif has_codechange:
                self._add_pull_request_comment(pr, message)
        if not pr:
            pr = self._create_new_pull_request(
                title, message, head=branch, base=base_branch
            )

        return pr

    def update_e2e_test_branch_and_update_pr(
        self, branch: str
    ) -> Optional[PullRequest]:
        base_branch = branch + "_base"
        branch_name = branch + "_test"

        self._update_pr_base_branch(base_branch)

        # Now that we have an updated base branch, create a dummy commit on top
        # so that we can actually create a pull request (this path tests
        # upstream directly, without any mailbox patches applied).
        self.repo_local.git.checkout("-B", branch_name)
        self.repo_local.git.commit("--allow-empty", "--message", "Dummy commit")

        title = f"[test] {branch_name}"

        # Force push only if there is no branch or code changes.
        pushed = False
        if branch_name not in self.branches or self.repo_local.git.diff(
            branch_name, f"remotes/origin/{branch_name}"
        ):
            self.repo_local.git.push("--force", "origin", branch_name)
            pushed = True

        self._update_e2e_pr(title, base_branch, branch_name, pushed)

    def do_sync(self) -> None:
        # fetch most recent upstream
        if UPSTREAM_REMOTE_NAME in [x.name for x in self.repo_local.remotes]:
            urls = list(self.repo_local.remote(UPSTREAM_REMOTE_NAME).urls)
            if urls != [self.upstream_url]:
                logger.warning(f"remote upstream set to track {urls}, re-creating")
                self.repo_local.delete_remote(UPSTREAM_REMOTE_NAME)
                self.repo_local.create_remote(UPSTREAM_REMOTE_NAME, self.upstream_url)
        else:
            self.repo_local.create_remote(UPSTREAM_REMOTE_NAME, self.upstream_url)
        upstream_repo = self.repo_local.remote(UPSTREAM_REMOTE_NAME)
        upstream_repo.fetch(self.upstream_branch)
        upstream_branch = getattr(upstream_repo.refs, self.upstream_branch)
        _reset_repo(self.repo_local, f"{UPSTREAM_REMOTE_NAME}/{self.upstream_branch}")
        self.repo_local.git.push(
            "--force", "origin", f"{upstream_branch}:refs/heads/{self.repo_branch}"
        )
        self.upstream_sha = upstream_branch.object.hexsha

    def full_sync(self, path: str, url: str, branch: str) -> git.Repo:
        logging.info(f"Doing full clone from {redact_url(url)}, branch: {branch}")

        with HistogramMetricTimer(git_clone_duration, {"branch": branch}):
            shutil.rmtree(path, ignore_errors=True)
            repo = git.Repo.clone_from(url, path)
            _reset_repo(repo, f"origin/{branch}")

        git_clone_counter.add(1, {"branch": branch})
        return repo

    def fetch_repo(self, path: str, url: str, branch: str) -> git.Repo:
        logging.info(f"Checking local sync repo at {path}")

        if os.path.exists(f"{path}/.git"):
            repo = git.Repo.init(path)
            try:
                with HistogramMetricTimer(git_fetch_duration, {"branch": branch}):
                    # Update origin URL to support GH app token refreshes
                    repo.remote(name="origin").set_url(url)
                    repo.git.fetch("--prune", "origin")
                    _reset_repo(repo, f"origin/{branch}")

                git_fetch_counter.add(1)
                return repo
            except git.exc.GitCommandError:
                logger.exception("Exception fetching repo, falling back to full_sync")

        return self.full_sync(path, url, branch)

    def fetch_repo_branch(self) -> None:
        """
        Fetch the repository branch of interest only once
        """
        self.repo_local = self.fetch_repo(
            self.repo_dir, self.repo_url, self.repo_branch
        )
        ci_repo_local = self.fetch_repo(
            self.ci_repo_dir,
            self.ci_repo_url,
            self.ci_branch,
        )
        ci_repo_local.git.checkout(f"origin/{self.ci_branch}")

    def _update_pr_base_branch(self, base_branch: str):
        """
        Update the pull request base branch by resetting it to upstream state
        and then adding CI files to it.
        """
        # Basically, on a detached head representing upstream state, add CI
        # files. This is the state that the base branch should have. Then see if
        # that is already the state that it has in the remote repository. If
        # not, push it. Lastly, always make sure to check out `base_branch`.
        _reset_repo(self.repo_local, f"{UPSTREAM_REMOTE_NAME}/{self.upstream_branch}")
        self._add_ci_files()

        try:
            diff = self.repo_local.git.diff(f"remotes/origin/{base_branch}")
        except git.exc.GitCommandError:
            # The remote may not exist, in which case we want to push.
            diff = True

        if diff:
            self.repo_local.git.checkout("-B", base_branch)
            self.repo_local.git.push("--force", "origin", f"refs/heads/{base_branch}")
        else:
            self.repo_local.git.checkout("-B", base_branch, f"origin/{base_branch}")

    def _create_dummy_commit(self, branch_name: str) -> None:
        """
        Reset branch, create dummy commit
        """
        _reset_repo(self.repo_local, f"{UPSTREAM_REMOTE_NAME}/{self.upstream_branch}")
        self.repo_local.git.checkout("-B", branch_name)
        self.repo_local.git.commit("--allow-empty", "--message", "Dummy commit")
        self.repo_local.git.push("--force", "origin", branch_name)

    def _close_pr(self, pr: PullRequest) -> None:
        pr.edit(state="closed")

    async def _guess_pr(
        self, series: Series, branch: Optional[str] = None
    ) -> Optional[PullRequest]:
        """
        Series could change name
        first series in a subject could be changed as well
        so we want to
        - try to guess based on name first
        - try to guess based on first series
        """

        # try to find amond known relevant PRs
        if series.subject in self.prs:
            return self.prs[series.subject]

        if not branch:
            # resolve branch: series -> subject -> branch
            subject = Subject(series.subject, self.patchwork)
            branch = await self.subject_to_branch(subject)

        try:
            # we assuming only one PR can be active for one head->base
            return self.all_prs[branch][self.repo_branch][0]
        except (KeyError, IndexError):
            pass

        # we failed to find active PR, now let's try to guess closed PR
        # is:pr is:closed head:"series/358111=>bpf"
        return self.filter_closed_pr(branch)

    async def _comment_series_pr(
        self,
        series: Series,
        branch_name: str,
        message: Optional[str] = None,
        can_create: bool = False,
        close: bool = False,
        has_merge_conflict: bool = False,
    ) -> Optional[PullRequest]:
        """
        Appends comment to a PR.
        """
        title = f"{series.subject}"
        tags = await series.visible_tags()
        pr_labels = copy.copy(tags)
        pr_labels.add(self.repo_branch)

        if has_merge_conflict:
            pr_labels.add(MERGE_CONFLICT_LABEL)

        pr = await self._guess_pr(series, branch=branch_name)

        if pr and pr.state == "closed":
            if can_create:
                try:
                    logger.info(f"Trying to re-open {pr}")
                    pr.edit(state="open")
                    self.add_pr(pr)
                    self.prs[pr.title] = pr
                except GithubException:
                    logger.warning(
                        f"Error re-opening {pr}, treating PR as non-exists.",
                        exc_info=True,
                    )
                    pr = None
            elif close:
                # we closing PR and it's already closed
                return pr

        if not pr and can_create and not close:
            # If there is no merge conflict and no change, ignore the series
            if not has_merge_conflict and not self.repo_local.git.diff(
                self.repo_pr_base_branch, branch_name
            ):
                # raise an exception so it bubbles up to the caller.
                raise NewPRWithNoChangeException(self.repo_pr_base_branch, branch_name)
            # we creating new PR
            logger.info(f"Creating PR for '{series.subject}' with {series.age()} delay")
            pr_created.add(1)
            if has_merge_conflict:
                pr_merge_conflict.add(1)
                self._create_dummy_commit(branch_name)

            pr = self._create_new_pull_request(
                title=title,
                message=(
                    f"Pull request for series with\n"
                    f"subject: {title}\n"
                    f"version: {series.version}\n"
                    f"url: {series.web_url}\n"
                ),
                head=branch_name,
                base=self.repo_pr_base_branch,
            )
        elif not pr and not can_create:
            # we closing PR and it's not found
            # how we get onto this state? expired and closed filtered on PW side
            # if we got here we already got series
            # this potentially indicates a bug in PR <-> series mapping
            # or some weird condition
            # this also may happen when we trying to add labels
            errors.add(1, {"msg": "missing_pull_request"})
            logger.error(f"BUG: Unable to find PR for {title} {series.web_url}")
            return None

        if pr:
            if (not has_merge_conflict) or (
                has_merge_conflict and not _is_pr_flagged(pr)
            ):
                if message:
                    self._add_pull_request_comment(pr, message)
                    pr_updated.add(1)

            # Make sure that we preserve any CI status labels.
            status_labels = {
                suffix.to_label(series.version) for suffix in StatusLabelSuffixes
            }
            labels = {label.name for label in pr.labels if label.name in status_labels}
            pr.set_labels(*pr_labels | labels)

            if close:
                pr_closed.add(1)
                if await series.is_expired():
                    pr_closed.add(1, {"reason": "expired"})
                logger.warning(f"Closing {pr}: {pr.head.ref}")
                self._close_pr(pr)
        return pr

    async def _pr_closed(self, branch_name: str, series: Series) -> bool:
        if await series.is_closed():
            warn_msg = f"At least one diff in series {series.web_url} irrelevant now. Closing PR."
        elif await series.is_expired():
            warn_msg = (
                f"At least one diff in series {series.web_url} expired. Closing PR."
            )
        elif not await series.has_matching_patches():
            warn_msg = (
                f"At least one diff in series {series.web_url} irrelevant now "
                f"for {self.patchwork.search_patterns} search patterns"
            )
        else:
            return False

        logger.warning(warn_msg)
        await self._comment_series_pr(
            series, message=warn_msg, close=True, branch_name=branch_name
        )

        # delete branch if there is no more PRs left from this branch
        prs = self.all_prs.get(branch_name, [])
        if await series.is_closed() and len(prs) == 1 and branch_name in self.branches:
            self.delete_branch(branch_name)

        return True

    def delete_branch(self, branch_name: str) -> None:
        logger.warning(f"Removing branch {branch_name}")
        branch_deleted.add(1)
        self.repo.get_git_ref(f"heads/{branch_name}").delete()

    def _add_ci_files(self) -> None:
        """
        Copy over and commit CI files (from the CI repository) to the current
        local repository's currently checked out branch.
        """
        if Path(f"{self.ci_repo_dir}/.github").exists():
            execute_command(f"cp --archive {self.ci_repo_dir}/.github {self.repo_dir}")
            self.repo_local.git.add("--force", ".github")
        execute_command(f"cp --archive {self.ci_repo_dir}/* {self.repo_dir}")
        self.repo_local.git.add("--all", "--force")
        self.repo_local.git.commit("--all", "--message", "adding ci files")

    async def try_apply_mailbox_series(
        self, branch_name: str, series: Series
    ) -> Tuple[bool, Optional[Exception], Optional[Any]]:
        """Try to apply a mailbox series and return (True, None, None) if successful"""
        # The pull request will be created against `repo_pr_base_branch`. So
        # prepare it for that.
        self._update_pr_base_branch(self.repo_pr_base_branch)
        self.repo_local.git.checkout("-B", branch_name)

        # Apply series
        patch_content = await series.get_patch_binary_content()
        with temporary_patch_file(patch_content) as tmp_patch_file:
            try:
                self.repo_local.git.am("--3way", istream=tmp_patch_file)
            except git.exc.GitCommandError as e:
                logger.warning(
                    f"Failed complete 3-way merge series {series.id} patch into {branch_name} branch: {e}"
                )
                conflict = self.repo_local.git.diff()
                return (False, e, conflict)
        return (True, None, None)

    async def apply_push_comment(
        self, branch_name: str, series: Series
    ) -> Optional[PullRequest]:
        comment = (
            f"Upstream branch: {self.upstream_sha}\nseries: {series.web_url}\n"
            f"version: {series.version}\n"
        )
        success, e, conflict = await self.try_apply_mailbox_series(branch_name, series)
        if not success:
            # The upstream git repo could have raced with patchwork.
            #
            # In other words, patchwork could be reporting a relevant
            # status (ie. !accepted) while the series has already been
            # merged and pushed.
            if await _series_already_applied(self.repo_local, series):
                logger.info(f"Series {series.url} already applied to tree")
                raise NewPRWithNoChangeException(self.repo_pr_base_branch, branch_name)

            comment = (
                f"{comment}\nPull request is *NOT* updated. Failed to apply {series.web_url}\n"
                f"error message:\n```\n{e}\n```\n\n"
                f"conflict:\n```\n{conflict}\n```\n"
            )
            logger.warning(f"Failed to apply {series.url}")
            pr_merge_conflict.add(1)
            return await self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                has_merge_conflict=True,
                can_create=True,
            )
        # force push only if if's a new branch or there is code or metadata diffs between old and new branches
        # which could mean that we applied new set of patches or just rebased
        if branch_name in self.branches and (
            branch_name not in self.all_prs  # NO PR yet
            or _is_branch_changed(
                self.repo_local,
                f"remotes/origin/{self.repo_pr_base_branch}",
                f"remotes/origin/{branch_name}",
                branch_name,
            )  # have code changes
        ):
            # we have branch, but either NO PR or there is code changes, we must try to
            # re-open PR first, before doing force-push.
            pr = await self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                can_create=True,
            )
            assert pr
            self.repo_local.git.push("--force", "origin", branch_name)

            # Metadata inside `pr` may be stale from the force push; refresh it
            pr.update()
            wanted_sha = self.repo_local.head.commit.hexsha
            for _ in range(30):
                if pr.head.sha == wanted_sha:
                    break
                logger.info(f"Waiting for {pr} sha={pr.head.sha} to go to {wanted_sha}")
                await asyncio.sleep(1)
                pr.update()
            else:
                raise RuntimeError("Github failed to update PR after force push")

            return pr
        # we don't have a branch, also means no PR, push first then create PR
        elif branch_name not in self.branches:
            if not self.repo_local.git.diff(self.repo_pr_base_branch, branch_name):
                # raise an exception so it bubbles up to the caller.
                raise NewPRWithNoChangeException(self.repo_pr_base_branch, branch_name)
            self.repo_local.git.push("--force", "origin", branch_name)
            return await self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                can_create=True,
            )
        else:
            # no code changes, just update labels
            return await self._comment_series_pr(series, branch_name=branch_name)

    async def checkout_and_patch(
        self, branch_name: str, series_to_apply: Series
    ) -> Optional[PullRequest]:
        """
        Patch in place and push.
        Returns true if whole series applied.
        Return None if at least one patch in series failed.
        Raises NewPRWithNoChangeException if series would not result in any changes.
        If at least one patch in series failed nothing gets pushed.
        """
        if await self._pr_closed(branch_name, series_to_apply):
            return None
        return await self.apply_push_comment(branch_name, series_to_apply)

    def add_pr(self, pr: PullRequest) -> None:
        self.all_prs.setdefault(pr.head.ref, {}).setdefault(pr.base.ref, [])
        self.all_prs[pr.head.ref][pr.base.ref].append(pr)
        logger.info(f"Found/tracking PR {pr.title=}, {pr.head.ref=}, {pr.base.ref=}")

    def get_pulls(self) -> None:
        self.prs = {}
        for pr in self.repo.get_pulls(state="open", base=self.repo_pr_base_branch):
            if self._is_relevant_pr(pr):
                self.prs[pr.title] = pr

            # This check is probably redundant given that we are filtering for open PRs only already.
            if pr.state == "open":
                self.add_pr(pr)

    def _is_relevant_pr(self, pr: PullRequest) -> bool:
        """
        PR is relevant if it
        - coming from user
        - to same user
        - to branch {repo_branch}
        - is open
        """
        src_user = none_throws(pr.head.user).login
        tgt_user = none_throws(pr.base.user).login
        pr_user = none_throws(pr.user).login
        tgt_branch = pr.base.ref
        state = pr.state
        if (
            src_user == self.user_or_org
            and tgt_user == self.user_or_org
            and pr_user == self.user_login
            and tgt_branch == self.repo_pr_base_branch
            and state == "open"
        ):
            return True
        return False

    def closed_prs(self) -> List[Any]:
        # GH api is not working: https://github.community/t/is-api-head-filter-even-working/135530
        # so i have to implement local cache
        # and local search
        # closed prs are last resort to re-open expired PRs
        # and also required for branch expiration
        if not self._closed_prs:
            self._closed_prs = list(
                self.repo.get_pulls(state="closed", base=self.repo_pr_base_branch)
            )
        return self._closed_prs

    def filter_closed_pr(self, head: str) -> Optional[PullRequest]:
        # this assumes only the most recent one closed PR per head
        res = None
        for pr in self.closed_prs():
            if pr.head.ref == head and (
                not res or res.updated_at.timestamp() < pr.updated_at.timestamp()
            ):
                res = pr
        return res

    async def subject_to_branch(self, subject: Subject) -> str:
        return f"{await subject.branch}{HEAD_BASE_SEPARATOR}{self.repo_branch}"

    async def sync_checks(self, pr: PullRequest, series: Series) -> None:
        # Make sure that we are working with up-to-date data (as opposed to
        # cached state).
        pr.update()
        # if it's merge conflict - report failure
        ctx = BranchWorker.slugify_context(f"{CI_DESCRIPTION}-{self.repo_branch}")
        if _is_pr_flagged(pr):
            await series.set_check(
                status=Status.CONFLICT,
                target_url=pr.html_url,
                context=f"{ctx}-PR",
                description=MERGE_CONFLICT_LABEL,
            )
            await self.evaluate_ci_result(Status.CONFLICT, series, pr, [])
            return

        logger.info(f"Fetching workflow runs for {pr}: {pr.head.ref} (@ {pr.head.sha})")

        statuses: List[Status] = []
        jobs = []

        # Note that we are interested in listing *all* runs and not just, say,
        # completed ones. The reason being that the information that pending
        # ones are present is very much relevant for status reporting.
        for run in self.repo.get_workflow_runs(
            actor=self.user_login,
            head_sha=pr.head.sha,
        ):
            status = gh_conclusion_to_status(run.conclusion)
            run_jobs = run.jobs()

            # Overall run failure could have many reasons, including
            # infrastructure issues or an in-progress rebase. Make an attempt at
            # detecting those. To reduce the number of failures we report
            # prematurely (runs will be retried eventually).
            if status == Status.FAILURE:
                for job in run_jobs:
                    for step in job.steps:
                        # If a step has no conclusion but the overall run is
                        # failed, the workflow was likely interrupted
                        # prematurely and never got a chance to finish (e.g.,
                        # because the runner died). Do not report failure, as we
                        # shall retry eventually.
                        # Similarly, if KPD does a force push because the
                        # upstream baseline changed we map that to pending. The
                        # run will be retried.
                        if step.conclusion is None or step.conclusion == "cancelled":
                            logger.info(
                                f"Step {step.name} of {run} was interrupted/canceled; marking workflow as pending"
                            )
                            status = Status.PENDING
                            break

            statuses.append(status)
            jobs += run_jobs

        status = process_statuses(statuses)
        # In order to keep PW contexts somewhat deterministic, we sort the array
        # of jobs by name and later use the index of the test in the array to
        # generate the context name.
        jobs = sorted(jobs, key=lambda job: job.name)
        jobs_logs = [
            f"{job.conclusion} -> {gh_conclusion_to_status(job.conclusion)} ({job.html_url})"
            for job in jobs
        ]

        logger.info(f"Workflow status: overall: '{status}', jobs: '{jobs_logs}")
        tasks = [
            self.submit_pr_summary(
                series=series,
                status=status,
                context_name=ctx,
                target_url=pr.html_url,
            )
        ] + [
            series.set_check(
                status=gh_conclusion_to_status(job.conclusion),
                target_url=job.html_url,
                context=BranchWorker.slugify_context(f"{ctx}-{CI_VMTEST_NAME}-{idx}"),
                description=f"Logs for {job.name}",
            )
            for idx, job in enumerate(jobs)
        ]
        await asyncio.gather(*tasks)

        await self.evaluate_ci_result(status, series, pr, jobs)

    async def evaluate_ci_result(
        self, status: Status, series: Series, pr: PullRequest, jobs: List[WorkflowJob]
    ) -> None:
        """Evaluate the result of a CI run and send an email as necessary."""
        email = self.email
        if email is None:
            logger.info("No email configuration present; skipping sending...")
            return

        if status in (Status.PENDING, Status.SKIPPED):
            return

        if status == Status.SUCCESS:
            new_label = StatusLabelSuffixes.PASS.to_label(series.version)
            not_label = StatusLabelSuffixes.FAIL.to_label(series.version)
        else:
            assert status in (Status.FAILURE, Status.CONFLICT), status
            new_label = StatusLabelSuffixes.FAIL.to_label(series.version)
            not_label = StatusLabelSuffixes.PASS.to_label(series.version)

        labels = {label.name for label in pr.labels}
        # Always make sure to remove the unused label so that we eventually
        # converge on having only one pass/fail label for each version, come
        # whatever.
        if not_label in labels:
            logger.info(
                f"{pr} was previously {not_label} and "
                f"is now {new_label}; removing {not_label} label"
            )
            pr.remove_from_labels(not_label)

        if new_label not in labels:
            # Either this is the first run we had for this patch version (no
            # label was there) or we switched states (pass <-> fail). Either
            # way, send an email notifying the submitter.
            logger.info(f"{pr} is now {new_label}; adding label")
            pr.add_to_labels(new_label)

            logger.info(f"Sending email notification for {pr}")
            failed_logs = await self.log_extractor.extract_failed_logs(jobs)
            inline_logs = self.log_extractor.generate_inline_email_text(failed_logs)
            subject = await get_ci_email_subject(series)
            ctx = build_email_body_context(self.repo, pr, status, series, inline_logs)
            body = furnish_ci_email_body(ctx)
            await send_email(email, series, subject, body)
            bump_email_status_counters(status)

    def expire_branches(self) -> None:
        for branch in self.branches:
            # all branches
            if branch in self.all_prs:
                # that are not belong to any known open prs
                continue

            if HEAD_BASE_SEPARATOR in branch:
                split = branch.split(HEAD_BASE_SEPARATOR)
                if len(split) > 1 and split[1] == self.repo_branch:
                    # which have our repo_branch as target
                    # that doesn't have any closed PRs
                    # with last update within defined TTL
                    pr = self.filter_closed_pr(branch)
                    if not pr or time.time() - pr.updated_at.timestamp() > BRANCH_TTL:
                        self.delete_branch(branch)

    def expire_user_prs(self) -> None:
        """
        Close user-created (i.e., non KPD) pull requests that have not seen an update
        in a while.
        """
        for pr in self.repo.get_pulls(state="open"):
            # Anything not created by KPD is fair game for being closed.
            if pr.user.login != self.user_login and _is_outdated_pr(pr):
                logger.info(f"Pull request {pr} is found to be outdated")
                self._add_pull_request_comment(
                    pr,
                    "Automatically cleaning up stale PR; feel free to reopen if needed",
                )

                self._close_pr(pr)

    async def submit_pr_summary(
        self, series: Series, status: Status, context_name: str, target_url: str
    ) -> None:
        logger.info(f"Submiting PR summary for series {series.id}")
        await series.set_check(
            status=status,
            target_url=target_url,
            context=f"{context_name}-PR",
            description="PR summary",
        )
        pr_summary_report.add(1)
