#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import json
import logging
import os
import re
from subprocess import PIPE, Popen
from typing import Dict

from github import Github, GithubException
from github.Repository import Repository
from kernel_patches_daemon.config import KPDConfig
from kernel_patches_daemon.daemon import KernelPatchesDaemon

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource


logger: logging.Logger = logging.getLogger(__name__)


class ScriptMetricsExporter:
    def __init__(self, script: str) -> None:
        self.script = script

    def export(self, project: str, metrics: Dict) -> None:
        if os.path.isfile(self.script) and os.access(self.script, os.X_OK):
            p = Popen([self.script], stdout=PIPE, stdin=PIPE, stderr=PIPE)
            p.communicate(input=json.dumps(metrics).encode())


class PurgeAction:
    @staticmethod
    def get_repo(git: Github, project: str) -> Repository:
        repo_name = os.path.basename(project)
        try:
            user = git.get_user()
            repo = user.get_repo(repo_name)
        except GithubException:
            org = ""
            if "https://" not in project and "ssh://" not in project:
                org = project.split(":")[-1].split("/")[0]
            else:
                org = project.split("/")[-2]
            repo = git.get_organization(org).get_repo(repo_name)
        return repo

    @staticmethod
    def run(kpd_config: KPDConfig) -> None:
        for branch, branch_config in kpd_config.branches.items():
            git = Github(branch_config.github_oauth_token)
            repo = PurgeAction.get_repo(git, branch)
            refs_to_remove = [
                f"heads/{branch_name}"
                for branch_name in repo.get_branches()
                if re.match(r"series/[0-9]+.*", branch_name.name)
            ]
            logging.info(f"Removing references: {refs_to_remove}")
            for ref in refs_to_remove:
                repo.get_git_ref(ref).delete()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Starts kernel-patches daemon")
    parser.add_argument(
        "--config",
        default="~/.kernel-patches/config.json",
        help="Specify config location",
    )
    parser.add_argument(
        "--label-colors",
        default="~/.kernel-patches/labels.json",
        help="Specify label coloring config location.",
    )
    parser.add_argument(
        "--metric-logger",
        default="~/.kernel-patches/metric_logger.sh",
        help="Specify external scripts which stdin will be fed with metrics",
    )
    parser.add_argument(
        "--action",
        default="start",
        choices=["start", "purge"],
        help="Purge will kill all existing PRs and delete all branches",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",
    )

    args: argparse.Namespace = parse_args()
    cfg_file = os.path.expanduser(args.config)
    labels_file = os.path.expanduser(args.label_colors)
    metrics_logger_script = os.path.expanduser(args.metric_logger)
    script_metrics_logger = ScriptMetricsExporter(metrics_logger_script)

    meter_provider = MeterProvider(
        resource=Resource(attributes={"service_name": "kernel_patches_daemon"}),
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())],
    )
    metrics.set_meter_provider(meter_provider)

    kpd_config = KPDConfig.from_file(cfg_file)

    with open(labels_file) as f:
        labels_cfg = json.load(f)

    if args.action == "purge":
        try:
            PurgeAction.run(kpd_config=kpd_config)
            exit(0)
        except Exception:
            logger.exception("Failed to purge")
            exit(1)

    daemon = KernelPatchesDaemon(
        kpd_config=kpd_config,
        labels_cfg=labels_cfg,
        metrics_logger=script_metrics_logger.export,
    )
    daemon.start()
