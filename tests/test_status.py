# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import permutations

from kernel_patches_daemon.status import process_statuses, Status


class UtilsTestCase(unittest.TestCase):
    def test_status_pending(self) -> None:
        """Test the process_statuses() function for pending status."""
        statuses = [
            Status.PENDING,
            Status.SUCCESS,
            Status.SKIPPED,
        ]

        for perm in permutations(statuses):
            with self.subTest(permutation=perm):
                self.assertEqual(process_statuses(perm), Status.PENDING)

    def test_status_failure(self) -> None:
        """Test the process_statuses() function for failure status."""
        statuses = [
            Status.PENDING,
            Status.SUCCESS,
            Status.SKIPPED,
            Status.FAILURE,
        ]

        for perm in permutations(statuses):
            with self.subTest(permutation=perm):
                self.assertEqual(process_statuses(perm), Status.FAILURE)

    def test_status_success(self) -> None:
        """Test the process_statuses() function for success status."""
        statuses = [
            Status.SUCCESS,
            Status.SKIPPED,
        ]

        for perm in permutations(statuses):
            with self.subTest(permutation=perm):
                self.assertEqual(process_statuses(perm), Status.SUCCESS)
