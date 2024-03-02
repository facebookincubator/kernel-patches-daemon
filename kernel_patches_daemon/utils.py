# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from git.util import remove_password_if_present


def redact_url(url: str) -> str:
    """
    A URL may contain username/password. Replace the password so it does not end up in the logs.
    """
    return remove_password_if_present([url])[0]


def remove_unsafe_chars(s: str) -> str:
    """
    Remove "unsafe" characters, i.e., those that we may not want to preserve in file names.
    """
    return "".join([c for c in s if c in ["_", "-"] or (c.isalnum() and c.isascii())])
