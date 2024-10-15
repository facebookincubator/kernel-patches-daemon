# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import re
import unittest
from dataclasses import dataclass
from typing import Dict, List, Union
from unittest.mock import mock_open, patch

from kernel_patches_daemon.config import (
    BranchConfig,
    EmailConfig,
    GithubAppAuthConfig,
    InvalidConfig,
    KPDConfig,
    PatchworksConfig,
)


def read_fixture(filepath: str) -> dict[str, str | int | bool | dict]:
    with open(os.path.join(os.path.dirname(__file__), filepath)) as f:
        return json.load(f)


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_invalid_tag_to_branch_mapping(self) -> None:
        """
        Tests that if a tag is mapped to a branch that doesn't exist, an exception is raised.
        """
        # Load a valid config
        kpd_config_json = read_fixture("fixtures/kpd_config.json")

        @dataclass
        class TestCase:
            name: str
            tag_to_branch_mapping: dict[str, list[str]]

        test_cases = [
            TestCase(
                name="tag mapped to a non-existent branch",
                tag_to_branch_mapping={"tag": ["non_existent_branch"]},
            ),
            TestCase(
                name="tag mapped to some branches that exists and some that don't",
                tag_to_branch_mapping={
                    "tag": ["app_auth_key", "non_existent_branch", "oauth"]
                },
            ),
            TestCase(
                name="__DEFAULT__ mapped to a non-existent branch",
                tag_to_branch_mapping={"__DEFAULT__": ["non_existent_branch"]},
            ),
            TestCase(
                name="__DEFAULT__ mapped to some branches that exists and some that don't",
                tag_to_branch_mapping={
                    "__DEFAULT__": ["app_auth_key", "non_existent_branch", "oauth"]
                },
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                conf = kpd_config_json.copy()
                conf["tag_to_branch_mapping"] = case.tag_to_branch_mapping
                with self.assertRaises(InvalidConfig):
                    KPDConfig.from_json(conf)

    def test_valid_tag_to_branch_mapping(self) -> None:
        """
        Tests combinaisons of valid tag_to_branch_mapping setup.
        """
        # Load a valid config
        kpd_config_json = read_fixture("fixtures/kpd_config.json")

        @dataclass
        class TestCase:
            name: str
            tag_to_branch_mapping: dict[str, list[str]]

        test_cases = [
            TestCase(
                name="single tag mapped to a valid branch",
                tag_to_branch_mapping={"tag": ["oauth"]},
            ),
            TestCase(
                name="multiple tags mapped to valid branches",
                tag_to_branch_mapping={
                    "tag1": ["app_auth_key"],
                    "tag2": ["oauth"],
                },
            ),
            TestCase(
                name="multiple tags mapped to same branch",
                tag_to_branch_mapping={
                    "tag1": ["oauth"],
                    "tag2": ["oauth"],
                },
            ),
            TestCase(
                name="tags mapped to multiple valid branch",
                tag_to_branch_mapping={
                    "tag1": ["oauth", "app_auth_key"],
                    "tag2": ["oauth"],
                },
            ),
            TestCase(
                name="__DEFAULT__ mapped to a valid branch",
                tag_to_branch_mapping={"__DEFAULT__": ["app_auth_key"]},
            ),
            TestCase(
                name="__DEFAULT__ mapped to multiple valid branches",
                tag_to_branch_mapping={"__DEFAULT__": ["app_auth_key", "oauth"]},
            ),
            TestCase(
                name="tags and __DEFAULT__ mapped to multiple valid branches",
                tag_to_branch_mapping={
                    "tag1": ["app_auth_key"],
                    "tag2": ["app_auth_key", "oauth"],
                    "__DEFAULT__": ["app_auth_key"],
                },
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                conf = kpd_config_json.copy()
                conf["tag_to_branch_mapping"] = case.tag_to_branch_mapping
                KPDConfig.from_json(conf)

    def test_valid(self) -> None:
        kpd_config_json = read_fixture("fixtures/kpd_config.json")

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
            email=EmailConfig(
                smtp_host="mail.example.com",
                smtp_port=465,
                smtp_user="bot-bpf-ci",
                smtp_from="bot+bpf-ci@example.com",
                smtp_to=["email1-to@example.com", "email2-to@example.com"],
                smtp_cc=["email1-cc@example.com", "email2-cc@example.com"],
                smtp_pass="super-secret-is-king",
                smtp_http_proxy="http://example.com:8080",
                submitter_allowlist=[
                    re.compile("email1-allow@example.com"),
                    re.compile("email2-allow@example.com"),
                ],
                ignore_allowlist=True,
            ),
            tag_to_branch_mapping={"tag": ["app_auth_key_path"]},
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
