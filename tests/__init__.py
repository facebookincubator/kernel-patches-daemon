# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.resources
import logging

logging.disable()


def read_fixture(filepath: str) -> str:
    return (
        importlib.resources.files(__package__)
        .joinpath("fixtures")
        .joinpath(filepath)
        .read_text()
    )
