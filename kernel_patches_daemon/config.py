# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class UnsupportedConfigVersion(ValueError):
    def __init__(self, version: int) -> None:
        super().__init__(f"Unsupported config version {version}")


class InvalidConfig(ValueError):
    pass


@dataclass
class GithubAppAuthConfig:
    app_id: int
    installation_id: int
    private_key: str

    @classmethod
    def _read_private_key(cls, private_key_path: os.PathLike) -> str:
        try:
            with open(private_key_path) as f:
                return f.read()
        except OSError as e:
            raise InvalidConfig(
                f"Failed to read Github AppAuth private key {private_key_path}"
            ) from e

    @classmethod
    def from_json_v2(cls, json: Dict) -> "GithubAppAuthConfig":
        try:
            return cls(
                app_id=json["github_app_id"],
                installation_id=json["github_installation_id"],
                private_key=cls._read_private_key(json["github_private_key"]),
            )
        except KeyError as e:
            raise InvalidConfig(e)

    @classmethod
    def from_json_v3(cls, json: Dict) -> "GithubAppAuthConfig":
        private_key_config = json.keys() & {
            "private_key",
            "private_key_path",
        }
        if len(private_key_config) != 1:
            raise InvalidConfig(
                "Github AppAuth config expect to have private_key OR private_key_path"
            )

        private_key = json.get("private_key")
        if private_key_path := json.get("private_key_path"):
            private_key = cls._read_private_key(private_key_path)

        if not private_key:
            raise InvalidConfig("Failed to load Github AppAuth private key")

        try:
            return cls(
                app_id=json["app_id"],
                installation_id=json["installation_id"],
                private_key=private_key,
            )
        except KeyError as e:
            raise InvalidConfig(e)


@dataclass
class BranchConfig:
    repo: str
    upstream_repo: str
    upstream_branch: str
    ci_repo: str
    ci_branch: str
    github_oauth_token: Optional[str]
    github_app_auth: Optional[GithubAppAuthConfig]

    @classmethod
    def from_json_v2(cls, json: Dict) -> "BranchConfig":
        github_app_auth_config: Optional[GithubAppAuthConfig] = None
        try:
            github_app_auth_config = GithubAppAuthConfig.from_json_v2(json)
        except InvalidConfig as e:
            logger.warning(f"Failed to parse Github AppAuth config: {e} is missing")

        return cls(
            repo=json["repo"],
            upstream_repo=json["upstream"],
            upstream_branch=json.get("upstream_branch", "master"),
            ci_repo=json["ci_repo"],
            ci_branch=json["ci_branch"],
            github_oauth_token=json.get("github_oauth_token", None),
            github_app_auth=github_app_auth_config,
        )

    @classmethod
    def from_json_v3(cls, json: Dict) -> "BranchConfig":
        github_app_auth_config: Optional[GithubAppAuthConfig] = None
        if app_auth_json := json.get("github_app_auth"):
            try:
                github_app_auth_config = GithubAppAuthConfig.from_json_v3(app_auth_json)
            except InvalidConfig as e:
                logger.warning(f"Failed to parse Github AppAuth config: {e}")

        return cls(
            repo=json["repo"],
            upstream_repo=json["upstream"],
            upstream_branch=json.get("upstream_branch", "master"),
            ci_repo=json["ci_repo"],
            ci_branch=json["ci_branch"],
            github_oauth_token=json.get("github_oauth_token", None),
            github_app_auth=github_app_auth_config,
        )


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_from: str
    smtp_pass: str
    smtp_to: List[str]
    smtp_http_proxy: Optional[str]

    @classmethod
    def from_json_v2(cls, json: Dict) -> "EmailConfig":
        return cls(
            smtp_host=json["smtp_host"],
            smtp_port=json.get("smtp_port", 465),
            smtp_user=json["smtp_user"],
            smtp_from=json["smtp_from"],
            smtp_pass=json["smtp_pass"],
            smtp_to=json.get("smtp_to", []),
            smtp_http_proxy=json.get("smtp_http_proxy", None),
        )

    @classmethod
    def from_json_v3(cls, json: Dict) -> "EmailConfig":
        return cls(
            smtp_host=json["host"],
            smtp_port=json.get("port", 465),
            smtp_user=json["user"],
            smtp_from=json["from"],
            smtp_pass=json["pass"],
            smtp_to=json.get("to", []),
            smtp_http_proxy=json.get("http_proxy", None),
        )


@dataclass
class PatchworksConfig:
    base_url: str
    project: str
    search_patterns: List[Dict]
    lookback: int
    user: Optional[str]
    token: Optional[str]

    @classmethod
    def from_json_v2(cls, json: Dict) -> "PatchworksConfig":
        return cls(
            base_url=json.get("pw_server", urlparse(json["pw_url"]).netloc),
            project=json["project"],
            search_patterns=json["pw_search_patterns"],
            lookback=json["pw_lookback"],
            user=json.get("pw_user", None),
            token=json.get("pw_token", None),
        )

    @classmethod
    def from_json_v3(cls, json: Dict) -> "PatchworksConfig":
        return cls(
            base_url=json["server"],
            project=json["project"],
            search_patterns=json["search_patterns"],
            lookback=json["lookback"],
            user=json.get("api_username", None),
            token=json.get("api_token", None),
        )


@dataclass
class KPDConfig:
    version: int
    patchwork: PatchworksConfig
    email: Optional[EmailConfig]
    branches: Dict[str, BranchConfig]
    tag_to_branch_mapping: Dict[str, List[str]]
    base_directory: str

    @classmethod
    def from_json(cls, json: Dict) -> "KPDConfig":
        try:
            version = int(json["version"])
            if version == 2:
                return cls.from_json_v2(json)
            elif version == 3:
                return cls.from_json_v3(json)
            else:
                raise UnsupportedConfigVersion(version)
        except (KeyError, IndexError) as e:
            raise InvalidConfig("Invalid KPD config") from e

    @classmethod
    def from_json_v2(cls, json: Dict) -> "KPDConfig":
        email = None
        if (
            "smtp_host" in json
            and "smtp_user" in json
            and "smtp_from" in json
            and "smtp_pass" in json
        ):
            email = EmailConfig.from_json_v2(json)

        return cls(
            version=2,
            tag_to_branch_mapping=json.get("tag_to_branch_mapping", {}),
            patchwork=PatchworksConfig.from_json_v2(json),
            email=email,
            branches={
                name: BranchConfig.from_json_v2(json_config)
                for name, json_config in json["branches"].items()
            },
            base_directory=json.get("base_directory", "/tmp"),
        )

    @classmethod
    def from_json_v3(cls, json: Dict) -> "KPDConfig":
        return cls(
            version=3,
            tag_to_branch_mapping=json["tag_to_branch_mapping"],
            patchwork=PatchworksConfig.from_json_v3(json["patchwork"]),
            email=EmailConfig.from_json_v3(json["email"]) if "email" in json else None,
            branches={
                name: BranchConfig.from_json_v3(json_config)
                for name, json_config in json["branches"].items()
            },
            base_directory=json["base_directory"],
        )

    @classmethod
    def from_file(cls, path: os.PathLike) -> "KPDConfig":
        with open(path) as f:
            json_config = json.load(f)
            return cls.from_json(json_config)
