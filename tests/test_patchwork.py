# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import datetime
import re
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from aioresponses import aioresponses

from freezegun import freeze_time
from kernel_patches_daemon.patchwork import parse_tags, RELEVANT_STATES, Subject, TTL
from kernel_patches_daemon.status import Status
from multidict import MultiDict
from pyre_extensions import none_throws

from tests.common.patchwork_mock import (
    DEFAULT_CHECK_CTX,
    DEFAULT_CHECK_CTX_QUERY,
    DEFAULT_FREEZE_DATE,
    DEFAULT_TEST_RESPONSES,
    FOO_SERIES_FIRST,
    FOO_SERIES_LAST,
    get_default_pw_client,
    get_dict_key,
    init_pw_responses,
)


class PatchworkTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Base TestCase class that ensure any patchwork related test cases are properly initialized.
    """

    async def asyncSetUp(self) -> None:
        self._pw = get_default_pw_client()


class TestPatchwork(PatchworkTestCase):
    @aioresponses()
    async def test_get_wrapper(self, m) -> None:
        """
        Simple test to ensure that GET requests are properly mocked.
        """
        pattern = re.compile(r"^.*$")
        m.get(pattern, status=200, body=b"""{"key1": "value1", "key2": 2}""")

        resp = await self._pw._Patchwork__get("object")
        m.assert_called_once()

        json_resp = await resp.json()
        self.assertEqual(json_resp["key1"], "value1")

    @aioresponses()
    async def test_post_wrapper(self, m) -> None:
        """
        Simple test to ensure that POST requests are properly mocked.
        """
        pattern = re.compile(r"^.*$")
        m.post(pattern, status=200, body=b"""{"key1": "value1", "key2": 2}""")
        # Make sure user and token are set so the requests is actually posted.
        self._pw.pw_token = "1234567890"
        self._pw.pw_user = "somerandomuser"
        resp = await self._pw._Patchwork__post("some/random/url", "somerandomdata")
        m.assert_called_once()

        json_resp = await resp.json()
        self.assertEqual(json_resp["key1"], "value1")

    async def test_get_objects_recursive(self) -> None:
        url = "https://127.0.0.1:0/api/1.1/projects/"

        @dataclass
        class Response:
            body: bytes
            headers: Dict[str, str]
            url: str = "https://127.0.0.1:0/api/1.1/projects/"
            status_code: int = 200

        @dataclass
        class TestCase:
            name: str
            pages: List[Response]
            expected: List[Any]
            get_calls: int
            filters: Optional[Union[Dict[str, str], MultiDict]] = None

        test_cases = [
            TestCase(
                name="single page",
                pages=[Response(url=url, body=b'["a","b","c"]', headers={})],
                expected=["a", "b", "c"],
                get_calls=1,
            ),
            TestCase(
                name="Multiple pages with proper formatting",
                pages=[
                    Response(
                        url=url,
                        body=b'["a"]',
                        headers={
                            "Link": '<https://127.0.0.1:0/api/1.1/projects/?page=2>; rel="next"'
                        },
                    ),
                    Response(
                        url=url + "?page=2",
                        body=b'["b"]',
                        headers={
                            "Link": '<https://127.0.0.1:0/api/1.1/projects/?page=3>; rel="next", <https://127.0.0.1:0/api/1.1/projects/>; rel="prev"'
                        },
                    ),
                    Response(
                        url=url + "?page=3",
                        body=b'["c"]',
                        headers={
                            "Link": '<https://127.0.0.1:0/api/1.1/projects/?page=2>; rel="prev"'
                        },
                    ),
                ],
                expected=["a", "b", "c"],
                get_calls=3,
            ),
            TestCase(
                name="single page with single filters",
                pages=[
                    Response(
                        url=url + "?k1=v1&k2=v2", body=b'["a","b","c"]', headers={}
                    )
                ],
                expected=["a", "b", "c"],
                get_calls=1,
                filters={"k1": "v1", "k2": "v2"},
            ),
            TestCase(
                name="single page with list filters",
                pages=[
                    Response(
                        url=url + "?k1=v1&k2=v2&k2=v2.2",
                        body=b'["a","b","c"]',
                        headers={},
                    )
                ],
                expected=["a", "b", "c"],
                get_calls=1,
                filters=MultiDict([("k1", "v1"), ("k2", "v2"), ("k2", "v2.2")]),
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                with aioresponses() as m:
                    for resp in case.pages:
                        m.get(
                            resp.url,
                            status=resp.status_code,
                            headers=resp.headers,
                            body=resp.body,
                        )

                    resp = await self._pw._Patchwork__get_objects_recursive(
                        "projects", params=case.filters
                    )
                    self.assertEqual(resp, case.expected)
                    self.assertEqual(
                        sum([len(x) for x in m.requests.values()]), case.get_calls
                    )

    @aioresponses()
    async def test_try_post_nocred_nomutation(self, m) -> None:
        """
        When pw_token is not set or is an empty string, we will not call post.
        """
        pattern = re.compile(r"^.*$")
        m.post(pattern, status=200)
        self._pw.auth_token = None
        await self._pw._Patchwork__try_post(
            "https://127.0.0.1:0/some/random/url", "somerandomdata"
        )
        m.assert_not_called()

        self._pw.auth_token = ""
        await self._pw._Patchwork__try_post(
            "https://127.0.0.1:0/some/random/url", "somerandomdata"
        )
        m.assert_not_called()

    def test_format_since(self) -> None:
        @dataclass
        class TestCase:
            name: str
            now: str
            expected: str
            lookback: int = 3

        PDT_UTC_OFFSET = -7
        test_cases = [
            TestCase(
                name="1st of Jan 00:00 UTC",
                now="2010-01-01T00:00:00",
                expected="2009-12-29T00:00:00",
            ),
            TestCase(
                name="1st of Jan 00:00 PDT",
                now=f"2010-01-01T00:00:00{PDT_UTC_OFFSET:+03}:00",
                expected="2009-12-29T00:00:00",
            ),
            TestCase(
                name="1st of Jan 23:00 UTC",
                now="2010-01-01T23:00:00",
                expected="2009-12-29T00:00:00",
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.name):
                # Force local time to be PDT
                with freeze_time(case.now, tz_offset=PDT_UTC_OFFSET):
                    self.assertEqual(
                        self._pw.format_since(case.lookback),
                        case.expected,
                    )

    def test_parse_tags(self) -> None:
        @dataclass
        class TestCase:
            name: str
            title: str
            expected: Set[str]

        test_cases = [
            TestCase(
                name="title without tags",
                title="title",
                expected=set(),
            ),
            TestCase(
                name="title with all valid tags",
                title="[tag1, tag2]title",
                expected={"tag1", "tag2"},
            ),
            TestCase(
                name="title without ignorable tags",
                title="[tag, 1/18, v4]title",
                expected={"tag"},
            ),
            TestCase(
                name="tags with extra spaces",
                title="[tag3, tag1  , tag2]title",
                expected={"tag1", "tag2", "tag3"},
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.name):
                self.assertEqual(
                    parse_tags(case.title),
                    case.expected,
                )


class TestSeries(PatchworkTestCase):
    @aioresponses()
    async def test_series_closed(self, m) -> None:
        """
        If one of the patch is irrelevant, the series is closed.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(666)
        self.assertTrue(await series.is_closed())

    @aioresponses()
    async def test_series_not_closed(self, m) -> None:
        """
        If none of the patches are irrelevant, the series is not closed.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(665)
        self.assertFalse(await series.is_closed())

    @aioresponses()
    async def test_series_tags(self, m) -> None:
        """
        Series tags are extracted from the diffs/cover letter/serie's name. We extract the content
        from the square bracket content prefixing those names and filter some out.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(665)
        self.assertEqual(
            await series.all_tags(),
            {
                # cover letter tags
                "cover letter tag",
                "duplicate tag",
                # relevant states
                get_dict_key(RELEVANT_STATES),
                get_dict_key(RELEVANT_STATES, 1),
                # series version
                "V4",
                # Series name
                "a/b",
                # patches
                "first patch tag",
                "second patch tag",
                "some stuff with spaces",
            },
        )

    @aioresponses()
    async def test_series_visible_tags(self, m) -> None:
        """
        Series' visible tags are only taken from patch states and series version.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(665)
        self.assertEqual(
            await series.visible_tags(),
            {
                # relevant states
                get_dict_key(RELEVANT_STATES),
                get_dict_key(RELEVANT_STATES, 1),
                # series version
                "V4",
            },
        )

    @aioresponses()
    async def test_series_tags_handle_missing_values(self, m) -> None:
        """
        Test that we handle correctly series with empty cover_letter and/or no attaches patches.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(667)
        self.assertEqual(
            await series.all_tags(),
            {
                "V1",
            },
        )

    @aioresponses()
    async def test_series_is_expired(self, m) -> None:
        """
        Test that when we are passed expiration date, the series is reported expired.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(668)
        ttl = TTL[get_dict_key(TTL)]
        # take the date of the patch and move to 1 second after that.
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE) + datetime.timedelta(
            seconds=ttl + 1
        )
        with freeze_time(now):
            self.assertTrue(await series.is_expired())

    @aioresponses()
    async def test_series_is_not_expired(self, m) -> None:
        """
        Test that when we are not yet passed expiration date, the series is not reported expired.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)
        series = await self._pw.get_series_by_id(668)
        ttl = TTL[get_dict_key(TTL)]
        # take the date of the patch and move to 1 second before that.
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE) + datetime.timedelta(
            seconds=ttl - 1
        )
        with freeze_time(now):
            self.assertFalse(await series.is_expired())

    @aioresponses()
    async def test_get_latest_check_for_patch(self, m) -> None:
        """
        Tests `get_latest_check_for_patch` function,
        which now makes a GET request with additional queries to ensure that
        it's fetching single latest data with matching context value
        """
        # space intentional to test urlencoding
        CTX = "vmtest bpf-next-PR"
        # not using requests.utils.requote_uri() since that encodes space to %20, while requests.get() encodes space to +
        CTX_URLENCODED = "vmtest+bpf-next-PR"
        EXPECTED_ID = 42
        contexts_responses = {
            f"https://127.0.0.1:0/api/1.1/patches/12859379/checks/?context={CTX_URLENCODED}&order=-date&per_page=1": [
                {
                    "context": CTX,
                    "date": "2010-01-02T00:00:00",
                    "state": "pending",
                    "id": EXPECTED_ID,
                },
            ],
        }
        init_pw_responses(m, contexts_responses)

        check = await self._pw.get_latest_check_for_patch(12859379, CTX)
        self.assertEqual(check["id"], EXPECTED_ID)

    @aioresponses()
    async def test_series_checks_update_all_diffs(self, m) -> None:
        """
        Test that we update all the diffs in a serie if there is either no existing check
        or checks need update.
        """
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1:0/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                    },
                ],
                # Patches without existing context
                f"https://127.0.0.1:0/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
                f"https://127.0.0.1:0/api/1.1/patches/6653/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        init_pw_responses(m, contexts_responses)

        post_pattern = re.compile(r"^.*$")
        m.post(post_pattern, status=200, repeat=True)

        series = await self._pw.get_series_by_id(665)
        await series.set_check(
            context=DEFAULT_CHECK_CTX,
            status=Status.PENDING,
            target_url="https://127.0.0.1:0/target",
        )
        # Hack... aioresponses stores the requests that were made in a dictionary whose's key is a tuple,
        # of the form (method, url).
        # https://github.com/pnuckowski/aioresponses/blob/56b843319d5d0ae8a405f188e68d7ba8c7573bc8/aioresponses/core.py#L507
        self.assertEqual(len([x for x in m.requests.keys() if x[0] == "POST"]), 3)

    @aioresponses()
    async def test_series_checks_no_update_same_state_target(self, m) -> None:
        """
        Test that we don't update checks if the state and target_url have not changed.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1:0/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                        "target_url": TARGET_URL,
                    },
                ],
                # Patches without existing context
                f"https://127.0.0.1:0/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
                f"https://127.0.0.1:0/api/1.1/patches/6653/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        init_pw_responses(m, contexts_responses)

        post_pattern = re.compile(r"^.*$")
        m.post(post_pattern, status=200, repeat=True)

        series = await self._pw.get_series_by_id(665)
        await series.set_check(
            context=DEFAULT_CHECK_CTX, status=Status.PENDING, target_url=TARGET_URL
        )
        # First patch is not updates
        self.assertEqual(
            len([x for x in m.requests.keys() if x[0] == "POST"]),
            len(series.patches) - 1,
        )

    @aioresponses()
    async def test_series_checks_update_same_state_diff_target(self, m) -> None:
        """
        Test that we update checks if the state is the same, but target_url has changed.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1:0/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                        # target_url not matching
                        "target_url": TARGET_URL + "something",
                    },
                ],
                # Patches without existing context
                f"https://127.0.0.1:0/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
                f"https://127.0.0.1:0/api/1.1/patches/6653/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        init_pw_responses(m, contexts_responses)

        post_pattern = re.compile(r"^.*$")
        m.post(post_pattern, status=200, repeat=True)

        series = await self._pw.get_series_by_id(665)
        await series.set_check(
            context=DEFAULT_CHECK_CTX, status=Status.PENDING, target_url=TARGET_URL
        )
        # First patch is not updates
        self.assertEqual(
            len([x for x in m.requests.keys() if x[0] == "POST"]), len(series.patches)
        )

    @aioresponses()
    async def test_series_checks_update_diff_state_same_target(self, m) -> None:
        """
        Test that we update checks if the state is not the same, but target_url is.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1:0/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                        "target_url": TARGET_URL,
                    },
                    {
                        "context": "other context",
                    },
                ],
                # Patches without existing context
                f"https://127.0.0.1:0/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
                f"https://127.0.0.1:0/api/1.1/patches/6653/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        init_pw_responses(m, contexts_responses)

        post_pattern = re.compile(r"^.*$")
        m.post(post_pattern, status=200, repeat=True)

        series = await self._pw.get_series_by_id(665)
        # success is a conclusive (non-pending) state.
        await series.set_check(
            context=DEFAULT_CHECK_CTX, status=Status.SUCCESS, target_url=TARGET_URL
        )
        # First patch is not updates
        self.assertEqual(
            len([x for x in m.requests.keys() if x[0] == "POST"]), len(series.patches)
        )

    @aioresponses()
    async def test_series_checks_no_update_diff_pending_state(self, m) -> None:
        """
        Test that we do not update checks if the new state is pending and we have an existing final state.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1:0/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        # conclusive state.
                        "state": "success",
                        "target_url": TARGET_URL,
                    },
                ],
                # Patches without existing context
                f"https://127.0.0.1:0/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
                f"https://127.0.0.1:0/api/1.1/patches/6653/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        init_pw_responses(m, contexts_responses)

        post_pattern = re.compile(r"^.*$")
        m.post(post_pattern, status=200, repeat=True)

        series = await self._pw.get_series_by_id(665)
        await series.set_check(
            context=DEFAULT_CHECK_CTX, status=Status.PENDING, target_url=TARGET_URL
        )
        # First patch is not updates
        self.assertEqual(
            len([x for x in m.requests.keys() if x[0] == "POST"]),
            len(series.patches) - 1,
        )


class TestSubject(PatchworkTestCase):
    @freeze_time(DEFAULT_FREEZE_DATE)
    @aioresponses()
    async def test_relevant_series(self, m) -> None:
        """
        Test that we find the relevant series for a given subject.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)

        s = Subject("foo", self._pw)
        # There is 2 relevant series
        self.assertEqual(len(await s.relevant_series), 4)

        # Series are ordered from oldest to newest
        series = (await s.relevant_series)[0]
        self.assertEqual(series.id, FOO_SERIES_FIRST)
        # It has only 1 diff, diff 11
        self.assertEqual(len(await series.get_patches()), 1)
        self.assertEqual([patch["id"] for patch in await series.get_patches()], [11])

    @freeze_time(DEFAULT_FREEZE_DATE)
    @aioresponses()
    async def test_latest_series(self, m) -> None:
        """
        Test that latest_series only returns.... the latest serie.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)

        s = Subject("foo", self._pw)
        latest_series = none_throws(await s.latest_series)
        # It is Series with ID FOO_SERIES_LAST
        self.assertEqual(latest_series.id, FOO_SERIES_LAST)
        # and has 3 diffs
        self.assertEqual(len(await latest_series.get_patches()), 3)

    @freeze_time(DEFAULT_FREEZE_DATE)
    @aioresponses()
    async def test_branch_name(self, m) -> None:
        """
        Test that the branch name is made using the first series ID in the list of relevant series.
        """
        init_pw_responses(m, DEFAULT_TEST_RESPONSES)

        s = Subject("foo", self._pw)
        branch = await s.branch
        # It is Series with ID 4
        self.assertEqual(branch, f"series/{FOO_SERIES_FIRST}")
