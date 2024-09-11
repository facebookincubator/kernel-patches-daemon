# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import re
from typing import Any, Dict, Final

from aioresponses import aioresponses

from kernel_patches_daemon.patchwork import (
    IRRELEVANT_STATES,
    Patchwork,
    RELEVANT_STATES,
    TTL,
)

DEFAULT_CHECK_CTX: Final[str] = "some_context"
DEFAULT_CHECK_CTX_QUERY: Final[str] = (
    f"?context={DEFAULT_CHECK_CTX}&order=-date&per_page=1"
)
PROJECT: Final[int] = 1234
DELEGATE: Final[int] = 12345


class PatchworkMock(Patchwork):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Force patchwork to use loopback/port 0 for whatever network access
        # we fail to mock
        # kwargs["server"] = "https://127.0.0.1"
        super().__init__(*args, **kwargs)


def get_default_pw_client() -> PatchworkMock:
    return PatchworkMock(
        server="127.0.0.1",
        api_version="1.1",
        search_patterns=[{"archived": False, "project": PROJECT, "delegate": DELEGATE}],
        auth_token="mocktoken",
    )


def init_pw_responses(m: aioresponses, data: Dict[str, Any]) -> None:
    """
    Setup an aioresponses mock to return patchwork answers.

    It takes a dictionary as input that uses the called URL as key and contains the
    value that we would expect from converting the response's json to native python type.

    If the query look like a search (e.g has query parameters), and we don't find a hit. We
    will return an empty list.
    If the query is not a search, and there is no key hit, we will return a 404.

    Under the hood, the value is converted back to json, but by using native types when
    generating the content for unittests, it makes it either and natively handle by the
    tooling than if we were writing raw json blobs. For instance, we can benefit from the linter,
    we can use variables/constants, we can use comments to explain why a blob of data is used....
    """
    for url, content in data.items():
        m.get(url, status=200, payload=content, repeat=True)

    # If we don't have the key and there is no search parameters assume (e.g no `?`` in URL)
    # we are querying a specific URL and it is not found.
    # This is a very rudimentation proxy, if we need to get more clever, aioresponses can take a function
    # to callback to
    m.get(re.compile(r"^[^?]+$"), status=404, body=b'{"detail": "Not found."}')
    # if there is query parameter, assume search and return an empty search result.
    m.get(re.compile(r"^.*$"), status=200, body=b"[]")


def get_dict_key(d: Dict[str, Any], idx: int = 0) -> str:
    """
    Given a dictionary, get a list of keys and return the one a `idx`.
    """
    return list(d)[idx]


FOO_SERIES_FIRST = 2
FOO_SERIES_LAST = 10

DEFAULT_FREEZE_DATE = "2010-07-23T00:00:00"
DEFAULT_TEST_RESPONSES: Dict[str, Any] = {
    "https://127.0.0.1/api/1.1/series/?q=foo": [
        # Does not match the subject name
        {
            "id": 1,
            "name": "foo bar",
            "date": "2010-07-20T01:00:00",
            "patches": [{"id": 10}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches and has relevant diff and is neither the oldest, nor the newest serie. Appears before FOO_SERIES_FIRST to ensure sorting is performed.
        {
            "id": 6,
            "name": "foo",
            "date": "2010-07-20T01:00:00",
            "patches": [{"id": 11}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches and has relevant diff
        {
            "id": FOO_SERIES_FIRST,
            "name": "foo",
            "date": "2010-07-20T00:00:00",
            "patches": [{"id": 11}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches and has only non-relevant diffs
        {
            "id": 3,
            "name": "foo",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 12}, {"id": 13}, {"id": 14}, {"id": 15}, {"id": 16}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches and has one relevant diffs
        {
            "id": FOO_SERIES_LAST,
            "name": "foo",
            "date": "2010-07-21T01:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches and has one relevant diffs, not the most recent series, appears after FOO_SERIES_LAST to ensure sorting is performed.
        {
            "id": 4,
            "name": "foo",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches and has only non-relevant diffs
        {
            "id": 5,
            "name": "foo",
            "date": "2010-07-21T02:00:00",
            "patches": [{"id": 12}, {"id": 13}, {"id": 14}, {"id": 15}, {"id": 16}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
    ],
    # Multiple relevant series to test our guess_pr logic.
    "https://127.0.0.1/api/1.1/series/?q=barv2": [
        # Matches and has relevant diff.
        {
            "id": 6,
            "name": "barv2",
            "date": "2010-07-20T01:00:00",
            "patches": [{"id": 11}],
            "version": 1,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
        # Matches, has one relevant diffs, and is the most recent series.
        {
            "id": 9,
            "name": "[v2] barv2",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 2,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
    ],
    # Single relevant series to test our guess_pr logic.
    "https://127.0.0.1/api/1.1/series/?q=code": [
        # Matches, has one relevant diffs, and is the most recent series.
        {
            "id": 9,
            "name": "[v2] barv2",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 2,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "submitter": {"email": "a-user@example.com"},
            "submitter": {"email": "a-user@example.com"},
            "mbox": "https://example.com",
        },
    ],
    # Correct project and delegate
    "https://127.0.0.1/api/1.1/patches/11/": {
        "id": 11,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": False,
    },
    # wrong project
    "https://127.0.0.1/api/1.1/patches/12/": {
        "id": 12,
        "project": {"id": PROJECT + 1},
        "delegate": {"id": DELEGATE},
        "archived": False,
    },
    # Wrong delegate
    "https://127.0.0.1/api/1.1/patches/13/": {
        "id": 13,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE + 1},
        "archived": False,
    },
    # Correct project/delegate but archived
    "https://127.0.0.1/api/1.1/patches/14/": {
        "id": 14,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
    },
    # None project
    "https://127.0.0.1/api/1.1/patches/15/": {
        "id": 15,
        "project": None,
        "delegate": {"id": DELEGATE},
        "archived": False,
    },
    # None delegate
    "https://127.0.0.1/api/1.1/patches/16/": {
        "id": 16,
        "project": {"id": PROJECT},
        "delegate": None,
        "archived": False,
    },
    #####################
    # Series test cases #
    #####################
    # An open series, is a series that has no patch in irrelevant state.
    "https://127.0.0.1/api/1.1/series/665/": {
        "id": 665,
        "name": "[a/b] this series is *NOT* closed!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6651}, {"id": 6652}, {"id": 6653}],
        "cover_letter": {"name": "[cover letter tag, duplicate tag] cover letter name"},
        "version": 4,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "submitter": {"email": "a-user@example.com"},
        "mbox": "https://example.com",
    },
    # Patch in an relevant state.
    "https://127.0.0.1/api/1.1/patches/6651/": {
        "id": 6651,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES),
        # No tag in name.
        "name": "foo",
    },
    # Patch in a relevant state.
    "https://127.0.0.1/api/1.1/patches/6652/": {
        "id": 6652,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES, 1),
        # Multiple tags. 0/5 and 1/2 should be ignored.
        # Same for v42 and v24, and patch.
        # only "first patch tag", "second patch tag", "some stuff with spaces" should be valid.
        "name": "[0/5, 1/2 , v42, V24, first patch tag, second patch tag, patch , some stuff with spaces , patch] bar",
    },
    # Patch in an relevant state.
    "https://127.0.0.1/api/1.1/patches/6653/": {
        "id": 6653,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES),
        # Single tag, which is a duplicate from cover letter.
        "name": "[duplicate tag] foo",
    },
    # A closed series, is a series that has at least 1 patch in an irrelevant state.
    "https://127.0.0.1/api/1.1/series/666/": {
        "id": 666,
        "name": "this series is closed!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6661}, {"id": 6662}],
        "version": 0,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "submitter": {"email": "a-user@example.com"},
        "mbox": "https://example.com",
    },
    # Patch in an irrelevant state.
    "https://127.0.0.1/api/1.1/patches/6661/": {
        "id": 6661,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(IRRELEVANT_STATES),
    },
    # Patch in a relevant state.
    "https://127.0.0.1/api/1.1/patches/6662/": {
        "id": 6662,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES),
    },
    # Series with no cover letter and no patches.
    "https://127.0.0.1/api/1.1/series/667/": {
        "id": 667,
        "name": "this series has no cover letter!",
        "date": "2010-07-20T01:00:00",
        "patches": [],
        "cover_letter": None,
        "version": 1,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "submitter": {"email": "a-user@example.com"},
        "mbox": "https://example.com",
    },
    # Expiration test cases
    # Series with expirable patches.
    "https://127.0.0.1/api/1.1/series/668/": {
        "id": 668,
        "name": "this series has no cover letter!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6681}, {"id": 6682}, {"id": 6683}],
        "cover_letter": None,
        "version": 1,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "submitter": {"email": "a-user@example.com"},
        "mbox": "https://example.com",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6681/": {
        "id": 6681,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Patch in an expirable state.
    "https://127.0.0.1/api/1.1/patches/6682/": {
        "id": 6682,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(TTL),
        "date": DEFAULT_FREEZE_DATE,
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6683/": {
        "id": 6683,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Series with no expirable patches.
    "https://127.0.0.1/api/1.1/series/669/": {
        "id": 669,
        "name": "this series has no cover letter!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6691}, {"id": 6692}, {"id": 6693}],
        "cover_letter": None,
        "version": 1,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "submitter": {"email": "a-user@example.com"},
        "mbox": "https://example.com",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6691/": {
        "id": 6691,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6692/": {
        "id": 6692,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6693/": {
        "id": 6693,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
}
