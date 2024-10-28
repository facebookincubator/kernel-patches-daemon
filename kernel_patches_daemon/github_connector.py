# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
from datetime import timedelta
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from github import Auth, Github, GithubException, GithubIntegration
from pyre_extensions import none_throws

logger: logging.Logger = logging.getLogger(__name__)

# FIXME: monkey patch the refresh threshold of GH App access token to 30min.
# This monkey patching may silently fail if the variable was renamed so first
# let assert if that variable does not exist.
# Next, we change its value to 30min so we can ensure that whenever we have a token,
# it is valid for at least 30 min.
# The reason we are doing this is that when we use this token for git operations,
# we are essentially setting a URL with the token embedded in it when pulling/pushing
# to git.
# We could reset the token for every write operations (the repo is world readable), but there is
# chances we will miss some. Instead, it will be easier to set the origin URL on every sync,
# giving us `ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS` to perform the operations.
# While arbitrary, this should be enough. For the case where we do a full clone,
# this should only happen on fresh start, which comes with a token with a validity of
# 2 hours.

ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS = 30 * 60
TOKEN_REFRESH_THRESHOLD_TIMEDELTA = timedelta(
    seconds=ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS
)
assert hasattr(
    Auth, "TOKEN_REFRESH_THRESHOLD_TIMEDELTA"
), "Could not monkey patch TOKEN_REFRESH_THRESHOLD_TIMEDELTA, it may have changed upstream."
Auth.TOKEN_REFRESH_THRESHOLD_TIMEDELTA = TOKEN_REFRESH_THRESHOLD_TIMEDELTA

BOT_USER_LOGIN_SUFFIX = "[bot]"


class AuthType(Enum):
    APP_AUTH = 1
    OAUTH_TOKEN = 2
    UNKNOWN = 3


class GithubConnector:
    """
    Base class for fetching basic Github Repo information for
    fbcode.kernel.kernel_patches_daemon.github.source.github_sync and
    fbcode.kernel.kernel_patches_daemon.statcollector
    """

    def __init__(
        self,
        repo_url: str,
        github_oauth_token: Optional[str] = None,
        app_auth: Optional[Auth.AppInstallationAuth] = None,
        http_retries: Optional[int] = None,
    ) -> None:

        assert bool(github_oauth_token) ^ bool(
            app_auth
        ), "Only one of github_oauth_token or app_auth can be set"
        self.repo_name: str = os.path.basename(repo_url)
        self.base_repo_url: str = repo_url
        self.auth_type = AuthType.UNKNOWN

        # Default to GH app auth if provided, and fallback to token based authentication.
        auth = (
            app_auth
            if app_auth is not None
            else Auth.Token(none_throws(github_oauth_token))
        )
        self.git: Github = Github(
            auth=auth,
            retry=http_retries,
        )
        gh_user = self.git.get_user()
        if app_auth is None:
            self.auth_type = AuthType.OAUTH_TOKEN
            self.user_login = gh_user.login
            self.github_account_name = gh_user.login
        else:
            self.auth_type = AuthType.APP_AUTH
            app = GithubIntegration(
                auth=Auth.AppAuth(
                    app_id=app_auth.app_id, private_key=app_auth.private_key
                )
            ).get_app()
            self.github_account_name = app.name
            # Github appends '[bot]' suffix to the NamedUser
            # >>> pull.user
            # NamedUser(login="kernel-patches-daemon-bpf[bot]")
            self.user_login = self.github_account_name + BOT_USER_LOGIN_SUFFIX
            # Note:
            # It seems that for a given app, GH creates and associated user with the '[bot]' suffix.
            # We could fetch that user to rely on IDs vs names.
            # I have not been able to find any evidence of this in the GH docs, though, so it is unclear
            # that it will not break down the line.
            # >>> pr.user
            # NamedUser(login="kernel-patches-daemon-bpf[bot]")
            # >>> pr.user.id
            # 128435009
            # >>> app.id
            # 307795
            # >>> app.name
            # 'kernel-patches-daemon-bpf'
            # >>> g.get_user('kernel-patches-daemon-bpf[bot]').id
            # 128435009
            # >>> g.get_user('kernel-patches-daemon-bpf[bot]').name
            # >>> g.get_user('kernel-patches-daemon-bpf[bot]').login
            # 'kernel-patches-daemon-bpf[bot]'

        logging.info(
            f"Using User login {self.user_login}, Github Account name {self.github_account_name}"
        )

        self.user_or_org: str = self.user_login

        try:
            # When using app_auth, this will raise a GithubException
            self.repo = gh_user.get_repo(self.repo_name)
        except GithubException:
            # are we working under org repo?
            org = ""
            if "https://" not in repo_url and "ssh://" not in repo_url:
                org = repo_url.split(":")[-1].split("/")[0]
            else:
                org = repo_url.split("/")[-2]
            self.user_or_org = org
            self.repo = self.git.get_organization(org).get_repo(self.repo_name)

        assert (
            self.auth_type != AuthType.UNKNOWN
        ), "Auth type is still set to unknown... something is wrong."

    def __get_new_auth_token(self) -> str:
        # refresh token if needed
        # pyre-fixme[16]: `github.MainClass.Github` has no attribute `__requester`.
        gh_requester = self.git._Github__requester
        return gh_requester.auth.token

    def __get_repo_url_with_auth_token(self) -> str:
        auth_token = self.__get_new_auth_token()
        parsed_url = urlparse(self.base_repo_url)

        # If there's no username then we have plain URL without credentials
        # In this case we append the user login and auth token to netloc
        if not parsed_url.username:
            return parsed_url._replace(
                netloc=f"{self.user_login}:{auth_token}@{parsed_url.netloc}"
            ).geturl()

        parsed_url = parsed_url._replace(
            netloc=parsed_url.netloc.replace(parsed_url.username, self.user_login)
        )

        if parsed_url.password:
            parsed_url = parsed_url._replace(
                netloc=parsed_url.netloc.replace(parsed_url.password, auth_token)
            )

        return parsed_url.geturl()

    @property
    def repo_url(self) -> str:
        # When using app_auth, the URL needs to be periodically updated as the auth token expires
        if self.auth_type == AuthType.APP_AUTH:
            return self.__get_repo_url_with_auth_token()

        return self.base_repo_url
