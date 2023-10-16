# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Iterator


class Status(Enum):
    SKIPPED = "skipped"
    SUCCESS = "success"
    FAILURE = "failure"
    CONFLICT = "conflict"


def gh_conclusion_to_status(gh_conclusion: str) -> Status:
    """Translate a GitHub conclusion to our `Status` enum."""
    # See
    # https://docs.github.com/en/rest/checks/suites?apiVersion=2022-11-28#get-a-check-suite
    # for a list of conclusions.
    if gh_conclusion in (
        "failure",
        "timed_out",
        "action_required",
        "startup_failure",
        "stale",
    ):
        return Status.FAILURE

    # A "success" overwrites any skips, as the latter are effectively
    # neutral.
    if gh_conclusion in ("success",):
        return Status.SUCCESS

    return Status.SKIPPED


def process_statuses(statuses: Iterator[Status]) -> Status:
    """Boil down a set of `Status` objects into a single one."""
    final = Status.SKIPPED
    for status in statuses:
        if status == Status.FAILURE:
            # "failure" is sticky.
            return status
        elif status == Status.SUCCESS:
            final = status
        else:
            # We ignore anything classified as `Skipped`, as that's the
            # starting state and we treat it as "neutral".
            pass
    return final
