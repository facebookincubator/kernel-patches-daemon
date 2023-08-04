# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import unittest
from typing import Dict
from unittest.mock import mock_open, patch

from kernel_patches_daemon.config import (
    BranchConfig,
    GithubAppAuthConfig,
    KPDConfig,
    PatchworksConfig,
)


def read_fixture(filepath: str) -> Dict:
    with open(os.path.join(os.path.dirname(__file__), filepath)) as f:
        return json.load(f)


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_valid_v3(self) -> None:
        kpd_config_json = read_fixture("fixtures/kpd_config_v3.json")

        with patch(
            "builtins.open", mock_open(read_data="TEST_KEY_FILE_CONTENT")
        ) as mock_file:
            config = KPDConfig.from_json(kpd_config_json)
            mock_file.assert_called_with("/key.pem")

        expected_config = KPDConfig(
            version=3,
            patchwork=PatchworksConfig(
                base_url="patchwork.kernel.org",
                project="unittest",
                user="unittest_user",
                token="unittest_token",
                search_patterns=[{"key": "value"}],
                lookback=1,
            ),
            tag_to_branch_mapping={"tag": ["branch"]},
            branches={
                "app_auth_key": BranchConfig(
                    github_app_auth=GithubAppAuthConfig(
                        app_id=123, installation_id=456, private_key="TEST_KEY_CONTENT"
                    ),
                    repo="https://repo.git",
                    upstream_repo="https://upstream.git",
                    upstream_branch="upstream_branch",
                    ci_repo="https://cirepo.git",
                    ci_branch="ci_branch",
                    github_oauth_token=None,
                ),
                "app_auth_key_path": BranchConfig(
                    repo="https://repo.git",
                    upstream_repo="https://upstream.git",
                    upstream_branch="upstream_branch",
                    ci_repo="https://cirepo.git",
                    ci_branch="ci_branch",
                    github_app_auth=GithubAppAuthConfig(
                        app_id=123,
                        installation_id=456,
                        private_key="TEST_KEY_FILE_CONTENT",
                    ),
                    github_oauth_token=None,
                ),
                "oauth": BranchConfig(
                    repo="https://repo.git",
                    upstream_repo="https://upstream.git",
                    upstream_branch="upstream_branch",
                    ci_repo="https://cirepo.git",
                    ci_branch="ci_branch",
                    github_app_auth=None,
                    github_oauth_token="TEST_OAUTH_TOKEN",
                ),
            },
            base_directory="/repos",
        )
        self.assertEqual(config, expected_config)

    def test_valid_v2(self) -> None:
        kpd_config_json = read_fixture("fixtures/kpd_config_v2.json")

        with patch(
            "builtins.open", mock_open(read_data="TEST_KEY_FILE_CONTENT")
        ) as mock_file:
            config = KPDConfig.from_json(kpd_config_json)
            mock_file.assert_called_with("/key.pem")

        expected_config = KPDConfig(
            version=2,
            patchwork=PatchworksConfig(
                base_url="patchwork.kernel.org",
                project="unittest",
                user="unittest_user",
                token="unittest_token",
                search_patterns=[{"key": "value"}],
                lookback=1,
            ),
            tag_to_branch_mapping={"tag": ["branch"]},
            branches={
                "app_auth_key": BranchConfig(
                    repo="https://repo.git",
                    upstream_repo="https://upstream.git",
                    upstream_branch="upstream_branch",
                    ci_repo="https://cirepo.git",
                    ci_branch="ci_branch",
                    github_app_auth=GithubAppAuthConfig(
                        app_id=123,
                        installation_id=456,
                        private_key="TEST_KEY_FILE_CONTENT",
                    ),
                    github_oauth_token=None,
                ),
                "oauth": BranchConfig(
                    repo="https://repo.git",
                    upstream_repo="https://upstream.git",
                    upstream_branch="upstream_branch",
                    ci_repo="https://cirepo.git",
                    ci_branch="ci_branch",
                    github_app_auth=None,
                    github_oauth_token="TEST_OAUTH_TOKEN",
                ),
            },
            base_directory="/tmp",
        )
        self.assertEqual(config, expected_config)
