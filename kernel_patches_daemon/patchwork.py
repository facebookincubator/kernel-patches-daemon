# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import datetime
import json
import logging
import re
from collections.abc import Sequence
from functools import update_wrapper
from types import SimpleNamespace
from typing import Any, AnyStr, Final, Optional
from urllib.parse import urljoin

import aiohttp
import cachetools.keys

import dateutil.parser as dateparser
from aiohttp_retry import ExponentialRetry, RetryClient
from cachetools import TTLCache

from kernel_patches_daemon.status import Status
from multidict import MultiDict

from opentelemetry import metrics
from pyre_extensions import none_throws

DEFAULT_HTTP_RETRIES = 3

# when we want to push this patch through CI
RELEVANT_STATES: dict[str, int] = {
    "new": 1,
    "under-review": 2,
    "rfc": 5,
    "changes-requested": 7,
    "queued": 13,
    "needs_ack": 15,
}
RFC_TAG: str = "RFC"
# with these tags will be closed if no updates within TTL
TTL = {"changes-requested": 3600, "rfc": 3600}

# when we are not interested in this patch anymore
IRRELEVANT_STATES: dict[str, int] = {
    "rejected": 4,
    "accepted": 3,
    "not-applicable": 6,
    "superseded": 9,
    "awaiting-upstream": 8,
    "deferred": 10,
    "mainlined": 11,
    "handled-elsewhere": 17,
}

PW_CHECK_PENDING_STATES: dict[Status, str] = {
    Status.PENDING: "pending",
}

PW_CHECK_CONCLUSIVE_STATES: dict[Status, str] = {
    Status.SUCCESS: "success",
    Status.SKIPPED: "success",
    Status.FAILURE: "fail",
    Status.CONFLICT: "fail",
}


PW_CHECK_STATES: dict[Status, str] = {
    **PW_CHECK_PENDING_STATES,
    **PW_CHECK_CONCLUSIVE_STATES,
}

SUBJECT_REGEXP: Final[re.Pattern] = re.compile(r"(?P<header>\[[^\]]*\])? *(?P<name>.+)")
IGNORE_TAGS_REGEX: Final[re.Pattern] = re.compile(
    r"([0-9]+/[0-9]+|V[0-9]+)|patch", re.IGNORECASE
)
TAG_REGEXP: Final[re.Pattern] = re.compile(r"^(\[(?P<tags>[^]]*)\])*")
PATCH_FILTERING_PROPERTIES = {"project", "delegate"}

logger: logging.Logger = logging.getLogger(__name__)
meter: metrics.Meter = metrics.get_meter("patchwork")

api_requests: metrics.Counter = meter.create_counter(name="requests.status")
api_requests_time: metrics.Histogram = meter.create_histogram(
    name="requests.duration_ms"
)
err_tag_parsing_failures: metrics.Counter = meter.create_counter(
    name="errors.tag_parsing_failures"
)
err_malformed_series: metrics.Counter = meter.create_counter(
    name="errors.malformed_series"
)


## Request Tracing
class TraceContext(SimpleNamespace):
    """
    aiohttp supports tracing.
    https://docs.aiohttp.org/en/stable/client_advanced.html#aiohttp-client-tracing

    To measure the time spent on a request, we capture the start time and end time in the TraceContext class
    by using the handlers `on_request_start` and `on_request_end` respectively.
    """

    start: float | None = None
    end: float | None = None

    def elapsed(self) -> float:
        return none_throws(self.end) - none_throws(self.start)


async def log_response_content(
    response: aiohttp.ClientResponse, trace_ctx: TraceContext
) -> None:
    """
    Log the response content for debugging.
    This will log the content of any requests which is not `ok`, e.g status code > 400:
    https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientResponse.ok
    Or any requests when in debugging mode.
    """
    log_level = logging.DEBUG if response.ok else logging.ERROR

    try:
        content = json_pprint(await response.json())
    except (json.decoder.JSONDecodeError, aiohttp.ContentTypeError):
        payload = await response.read()
        content = f"<binary content: {len(payload)} bytes>"
    except Exception:
        logger.exception("Failed to handle Patchwork response")
        return

    logger.log(
        log_level,
        f"Patchwork {response.method} {response.url} "
        f"{response.status}, response: {content}",
    )


def json_pprint(obj: Any, indent: int = 2) -> str:
    """
    Return a string which is suitable to pretty print a JSON object.
    """
    return json.dumps(obj, indent=indent, sort_keys=True)


async def log_response_metrics(
    response: aiohttp.ClientResponse, trace_ctx: TraceContext
) -> None:
    """
    Log an API response metrics.
    HTTP methods like GET, POST and OPTIONS per ranges of status codes are counted,
    as well as requests elapsed time per method.
    """
    response_code_leading_num = response.status // 100

    if not response.method:
        logger.error("No http method founds in response, skipping metrics")
        return

    normalized_http_method = response.method.strip().lower()
    api_requests.add(
        1,
        {
            "http_code": f"{response_code_leading_num}xx",
            "http_method": normalized_http_method,
        },
    )
    api_requests_time.record(
        round(trace_ctx.elapsed() * 1000),  # Elapsed time in ms
        {"http_method": normalized_http_method},
    )


async def on_request_start(
    session: aiohttp.ClientSession,
    trace_ctx: TraceContext,
    params: aiohttp.TraceRequestStartParams,
) -> None:
    """
    Handler called when an HTTP request starts.
    We currently use this to capture the start time of the request.
    """
    trace_ctx.start = asyncio.get_event_loop().time()


async def on_request_end(
    session: aiohttp.ClientSession,
    trace_ctx: TraceContext,
    params: aiohttp.TraceRequestEndParams,
) -> None:
    """
    Handler called when an HTTP request finishes.
    We currently use this to capture the end time of the request and subsequently
    log the response content and metrics.
    """
    trace_ctx.end = asyncio.get_event_loop().time()
    await log_response_content(params.response, trace_ctx)
    await log_response_metrics(params.response, trace_ctx)


def time_since_secs(date: str) -> float:
    parsed_datetime = dateparser.parse(date)
    duration = datetime.datetime.utcnow() - parsed_datetime
    return duration.total_seconds()


def parse_tags(input: str) -> set[str]:
    # "[tag1 ,tag2]title" -> "tag1,tags" -> ["tag1", "tag2"]
    try:
        parsed_tags = none_throws(re.match(TAG_REGEXP, input)).group("tags").split(",")
        logging.debug(f"Parsed tags from '{input}' string: {parsed_tags}")
    except Exception:
        logger.warning(f"Can't parse tags from string '{input}'")
        err_tag_parsing_failures.add(1)
        return set()

    tags = [tag.strip() for tag in parsed_tags]
    filtered_tags = {tag for tag in tags if not re.match(IGNORE_TAGS_REGEX, tag)}
    logging.debug(f"Filtered tags from '{input}' string: {filtered_tags}")
    return filtered_tags


def parse_subject(input: str) -> str:
    try:
        logging.debug(f"Parsing subject name from '{input}' patch name")
        return none_throws(re.match(SUBJECT_REGEXP, input)).group("name")
    except Exception as ex:
        logger.error(f"Failed to parse subject from patch name '{input}': {ex}")
        return ""


# An asyncio compatible cachetools.cached method.
# https://github.com/tkem/cachetools/issues/137
# We may want to use asyncache, aiocache or the likes.
def cached(cache, key=cachetools.keys.hashkey, lock=None):
    """
    Decorator to wrap an async function with a memoizing callable that saves results in a cache.
    """

    def decorator(func):
        if cache is None:

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

        elif lock is None:

            async def wrapper(*args, **kwargs):
                k = key(*args, **kwargs)
                try:
                    return cache[k]
                except KeyError:
                    pass  # key not found
                v = await func(*args, **kwargs)
                try:
                    cache[k] = v
                except ValueError:
                    pass  # value too large
                return v

        else:

            async def wrapper(*args, **kwargs):
                k = key(*args, **kwargs)
                try:
                    with lock:
                        return cache[k]
                except KeyError:
                    pass  # key not found
                v = await func(*args, **kwargs)
                try:
                    with lock:
                        cache[k] = v
                except ValueError:
                    pass  # value too large
                return v

        return update_wrapper(wrapper, func)

    return decorator


class Subject:
    def __init__(self, subject: str, pw_client: "Patchwork") -> None:
        self.pw_client = pw_client
        self.subject = subject

    @property
    async def branch(self) -> str | None:
        relevant_series = await self.relevant_series
        if len(relevant_series) == 0:
            return None
        return f"series/{relevant_series[0].id}"

    @property
    async def latest_series(self) -> Optional["Series"]:
        relevant_series = await self.relevant_series
        if len(relevant_series) == 0:
            return None
        return relevant_series[-1]

    @property
    @cached(cache=TTLCache(maxsize=1, ttl=600))
    async def relevant_series(self) -> list["Series"]:
        """
        cache and return sorted list of relevant series
        where first element is first known version of same subject
        and last is the most recent
        """
        series_list = await self.pw_client.get_series(params={"q": self.subject})

        logging.debug(
            f"All series for '{self.subject}' subject: {json_pprint([s.to_json() for s in series_list])}"
        )
        # we using full text search which could give ambiguous results
        # so we must filter out irrelevant results
        relevant_series = [
            series
            for series in series_list
            if series.subject == self.subject and await series.has_matching_patches()
        ]
        # sort series by age desc,  so last series is the most recent one
        sorted_series = sorted(relevant_series, key=lambda x: x.age(), reverse=True)
        logging.debug(
            f"Sorted matching series for '{self.subject}' subject: {json_pprint([s.to_json() for s in sorted_series])}"
        )
        return sorted_series

    def to_json(self) -> str:
        return json.dumps({"subject": self.subject})

    def __repr__(self) -> str:
        return f"Subject({self.to_json()})"


class Series:
    def __init__(self, pw_client: "Patchwork", data: dict) -> None:
        self.pw_client = pw_client
        self.data = data
        self._patch_blob = None

        # We should be able to create object from a short version of series object from /patches/ endpoint
        # Docs: https://patchwork.readthedocs.io/en/latest/api/rest/schemas/v1.2/#get--api-1.2-patches-
        # {
        #     "id": 1,
        #     "url": "https://example.com",
        #     "web_url": "https://example.com",
        #     "name": "string",
        #     "date": "string",
        #     "version": 1,
        #     "mbox": "https://example.com"
        # }
        self.id = data["id"]
        self.name = data["name"]
        self.date = data["date"]
        self.url = data["url"]
        self.web_url = data["web_url"]
        self.version = data["version"]
        self._submitter_email = data["submitter"]["email"]
        self.mbox = data["mbox"]
        self.patches = data.get("patches", [])
        self.cover_letter = data.get("cover_letter")

        try:
            logging.debug(f"Parsing subject name from '{self.name}' series name")
            self.subject = none_throws(re.match(SUBJECT_REGEXP, data["name"])).group(
                "name"
            )
        except Exception as ex:
            raise ValueError(
                f"Failed to parse subject from series name '{data['name']}'"
            ) from ex

    def _is_patch_matching(self, patch: dict[str, Any]) -> bool:
        for pattern in self.pw_client.search_patterns:
            for prop_name, expected_value in pattern.items():
                if prop_name in PATCH_FILTERING_PROPERTIES:
                    try:
                        # these values can be None so we need to filter them out first
                        if not patch[prop_name]:
                            return False
                        if patch[prop_name]["id"] != expected_value:
                            return False
                    except KeyError:
                        return False
                elif patch[prop_name] != expected_value:
                    return False
        return True

    def age(self) -> float:
        return time_since_secs(self.date)

    @cached(cache=TTLCache(maxsize=1, ttl=600))
    async def get_patches(self) -> tuple[dict]:
        """
        Returns patches preserving original order
        for the most recent relevant series
        """
        tasks = [self.pw_client.get_patch_by_id(patch["id"]) for patch in self.patches]
        return await asyncio.gather(*tasks)

    async def is_closed(self) -> bool:
        """
        Series considered closed if at least one patch in this series
        is in irrelevant states
        """
        for patch in await self.get_patches():
            if patch["state"] in IRRELEVANT_STATES:
                return True
        return False

    @cached(cache=TTLCache(maxsize=1, ttl=120))
    async def all_tags(self) -> set[str]:
        """
        Tags fetched from series name, diffs and cover letter
        for most relevant series
        """
        tags = {f"V{self.version}"}

        for patch in await self.get_patches():
            tags |= parse_tags(patch["name"])
            tags.add(patch["state"])

        if self.cover_letter:
            tags |= parse_tags(self.cover_letter["name"])

        tags |= parse_tags(self.name)

        return tags

    async def visible_tags(self) -> set[str]:
        return {
            f"V{self.version}",
            *[diff["state"] for diff in await self.get_patches()],
        }

    @cached(cache=TTLCache(maxsize=1, ttl=120))
    async def patch_subjects(self) -> list[str]:
        """
        Returns an ordered list of all patch subjects (tags removed)
        """
        return [parse_subject(patch["name"]) for patch in await self.get_patches()]

    async def is_expired(self) -> bool:
        for diff in await self.get_patches():
            if diff["state"] in TTL:
                if time_since_secs(diff["date"]) >= TTL[diff["state"]]:
                    return True
        return False

    @cached(cache=TTLCache(maxsize=1, ttl=120))
    async def get_patch_binary_content(self) -> bytes:

        content = await self.pw_client.get_blob(self.mbox)
        logger.debug(
            f"Received patch mbox for series {self.id}, size: {len(content)} bytes"
        )
        return content

    async def has_matching_patches(self) -> bool:
        for patch in await self.get_patches():
            if self._is_patch_matching(patch):
                return True

        return False

    async def set_check(self, status: Status, **kwargs) -> None:
        tasks = [
            self.pw_client.post_check_for_patch_id(
                patch_id=patch["id"], status=status, check_data=kwargs
            )
            for patch in self.patches
        ]
        await asyncio.gather(*tasks)

    def to_json(self) -> str:
        json_keys = {
            "id",
            "name",
            "date",
            "url",
            "web_url",
            "version",
            "mbox",
            "patches",
            "cover_letter",
        }
        return json.dumps(
            {k: getattr(self, k) for k in json_keys if getattr(self, k, None)}
        )

    def __repr__(self) -> str:
        return f"Series({self.to_json()})"

    @property
    def submitter_email(self):
        """Retrieve the email address of the patch series submitter."""
        return self._submitter_email


class Patchwork:
    def __init__(
        self,
        server: str,
        search_patterns: list[dict[str, Any]],
        auth_token: str | None = None,
        lookback_in_days: int = 7,
        api_version: str = "1.2",
        http_retries: int = DEFAULT_HTTP_RETRIES,
    ) -> None:
        self.api_url = f"https://{server}/api/{api_version}/"
        self.auth_token = auth_token
        if not auth_token:
            logger.warning("Patchwork client runs in read-only mode")
        self.search_patterns = search_patterns
        self.since = self.format_since(lookback_in_days)
        # member variable initializations
        self.known_series: dict[int, Series] = {}
        self.known_subjects: dict[str, Subject] = {}

        # aiohttp's ClientSession needs to be initialized within an async function.
        # We will differ this initialization to a separate function and memoize it during first call.
        self.http_retries = http_retries
        self.http_session = None

    async def get_http_session(self) -> aiohttp.ClientSession:
        """
        Return a cached http session and initialize it if it doesn't exist.
        """
        if self.http_session is None:
            # Setup aiohttp client tracing so we can log metrics/requests
            # https://docs.aiohttp.org/en/stable/client_advanced.html#aiohttp-client-tracing
            trace_config = aiohttp.TraceConfig(trace_config_ctx_factory=TraceContext)
            # pyre-fixme[6]: In call `typing.MutableSequence.append`, for 1st positional argument, expected `_SignalCallback[TraceRequestStartParams]` but got `typing.Callable(on_request_start)[[Named(session, ClientSession), Named(trace_ctx, TraceContext), Named(params, TraceRequestStartParams)], Coroutine[typing.Any, typing.Any, None]]`.
            trace_config.on_request_start.append(on_request_start)
            # pyre-fixme[6]: In call `typing.MutableSequence.append`, for 1st positional argument, expected `_SignalCallback[TraceRequestEndParams]` but got `typing.Callable(on_request_end)[[Named(session, ClientSession), Named(trace_ctx, TraceContext), Named(params, TraceRequestEndParams)], Coroutine[typing.Any, typing.Any, None]]`.
            trace_config.on_request_end.append(on_request_end)
            client_session = aiohttp.ClientSession(
                trace_configs=[trace_config],
                # Read proxy from env var
                trust_env=True,
            )

            # Work around intermittent issues by adding some retry logic.
            retry_options = ExponentialRetry(attempts=self.http_retries)
            self.http_session = RetryClient(
                client_session=client_session, retry_options=retry_options
            )
        return self.http_session

    def format_since(self, pw_lookback: int) -> str:
        today = datetime.datetime.utcnow().date()
        lookback = today - datetime.timedelta(days=pw_lookback)
        return lookback.strftime("%Y-%m-%dT%H:%M:%S")

    async def __get(
        self, path: AnyStr, allow_redirects=True, **kwargs: dict
    ) -> aiohttp.ClientResponse:
        http_session = await self.get_http_session()
        resp = await http_session.get(
            # pyre-ignore
            urljoin(self.api_url, path),
            # pyre-ignore
            **kwargs,
        )
        return resp

    async def __get_object_by_id(self, object_type: str, object_id: int) -> dict:
        resp = await self.__get(f"{object_type}/{object_id}/")
        return await resp.json()

    async def __get_objects_recursive(
        self, object_type: str, params: dict | None = None
    ) -> list[dict]:
        items = []
        path = f"{object_type}/"
        if params is None:
            params = {}
        while True:
            response = await self.__get(path, params=params)
            items += await response.json()

            if "next" not in response.links:
                break

            # FIXME: the returned URL is a yarl.URL that contains the full hostname, path, query parameters.
            # Here all a sudden, path is changed from a "relative path" to the full URL. Luckily, urljoin, which we use
            # in `__get` deal with this and compute the right URL.
            path = str(response.links["next"]["url"])
        return items

    async def __post(self, path: AnyStr, data: dict) -> aiohttp.ClientResponse:
        http_session = await self.get_http_session()
        resp = await http_session.post(
            # pyre-ignore
            urljoin(self.api_url, path),
            headers={"Authorization": f"Token {self.auth_token}"},
            data=data,
        )
        return resp

    async def __try_post(
        self, path: AnyStr, data: dict
    ) -> aiohttp.ClientResponse | None:
        if not self.auth_token:
            logger.debug(f"Patchwork POST {path}: read-only mode, request ignored")
            logger.debug(f"Patchwork POST data: {json_pprint(data)}")
            return None

        return await self.__post(path, data)

    async def get_blob(self, url: AnyStr) -> bytes:
        resp = await self.__get(url, allow_redirects=True)
        return await resp.read()

    async def get_latest_check_for_patch(
        self,
        patch_id: int,
        context: str,
    ) -> dict[str, Any]:
        """
        Return dict of the latest check with matching context and patch_id if exists.
        Returns None if such check does not exist.
        """
        # get single latest one with matching context value
        # patchwork.kernel.org api ignores case for context query
        resp = await self.__get(
            f"patches/{patch_id}/checks/",
            params={
                "context": context,
                "order": "-date",
                "per_page": "1",
            },
        )
        checks = await resp.json()
        logger.debug(
            f"Received latest check for patch {patch_id} with context {context}: {json_pprint(checks)}"
        )

        if len(checks) == 0:
            raise ValueError(
                f"Patch {patch_id} doesn't have any checks with '{context}' context"
            )

        return checks[0]

    async def post_check_for_patch_id(
        self, patch_id: int, status: Status, check_data: dict[str, Any]
    ) -> aiohttp.ClientResponse | None:
        new_state = PW_CHECK_STATES.get(status, PW_CHECK_STATES[Status.PENDING])
        updated_check_data = {
            **check_data,
            "state": new_state,
        }
        logger.debug(
            f"Trying to update check for {patch_id} with a new content: {json_pprint(updated_check_data)}"
        )
        try:
            check = await self.get_latest_check_for_patch(
                patch_id, check_data["context"]
            )
            if (
                status in PW_CHECK_PENDING_STATES
                and PW_CHECK_PENDING_STATES[status] != check["state"]
            ):
                logger.info(
                    f"Not posting state update for patch {patch_id}: "
                    f"existing state '{check['state']}' can't be changed "
                    f"to pending state '{new_state}'"
                )
                return None

            if (
                check.get("state") == new_state
                and check.get("target_url") == check_data["target_url"]
            ):
                logger.debug(
                    f"Not posting state update for patch {patch_id}: previous state '{new_state}' and url are the same"
                )
                return None

            logger.info(
                f"Updating patch {patch_id} check, current state: '{check.get('state')}', new state: '{new_state}'"
            )
        except ValueError:
            logger.info(f"Setting patch {patch_id} check to '{new_state}' state")

        return await self.__try_post(
            f"patches/{patch_id}/checks/",
            data=updated_check_data,
        )

    async def get_series_by_id(self, series_id: int) -> Series:
        # fetches directly only if series is not available in local scope
        if series_id not in self.known_series:
            self.known_series[series_id] = Series(
                self, await self.__get_object_by_id("series", series_id)
            )

        return self.known_series[series_id]

    def get_subject_by_series(self, series: Series) -> Subject:
        # local cache for subjects
        if series.subject not in self.known_subjects:
            subject = Subject(series.subject, self)
            self.known_subjects[series.subject] = subject

        return self.known_subjects[series.subject]

    async def get_relevant_subjects(self) -> Sequence[Subject]:
        subjects = {}
        filtered_subjects = []
        self.known_series = {}
        self.known_subjects = {}

        for pattern in self.search_patterns:
            patch_filters = MultiDict(
                [
                    ("since", self.since),
                    ("archived", str(False)),
                    *[("state", val) for val in RELEVANT_STATES.values()],
                ]
            )
            patch_filters.update({k: str(v) for k, v in pattern.items()})
            logger.info(
                f"Searching for Patchwork patches that match the criteria: {patch_filters}"
            )
            all_patches = await self.__get_objects_recursive(
                "patches", params=patch_filters
            )

            series_ids = set()
            for patch in all_patches:
                for series_data in patch["series"]:
                    if not series_data.get("name"):
                        logger.error(f"Malformed series name in: {series_data}")
                        err_malformed_series.add(1)
                        continue

                    try:
                        series_id = int(series_data["id"])
                        series_ids.add(series_id)
                    except ValueError:
                        logger.error(f"Malformed series ID in: {series_data}")
                        err_malformed_series.add(1)
                        continue

            tasks = [self.get_series_by_id(series_id) for series_id in series_ids]
            all_series = await asyncio.gather(*tasks)

            for series in all_series:
                self.known_series[series.id] = series

                if series.subject not in subjects:
                    subjects[series.subject] = Subject(series.subject, self)
                    self.known_subjects[series.subject] = subjects[series.subject]

            logger.info(f"Total subjects found: {len(subjects)}")

            # async function used to fetch latest series for each subject concurrently
            async def fetch_latest_series(
                subject_name, subject_obj
            ) -> tuple[str, Series, Series | None]:
                return (subject_name, subject_obj, await subject_obj.latest_series)

            tasks = [fetch_latest_series(k, v) for k, v in subjects.items()]
            tasks = await asyncio.gather(*tasks)

            for subject_name, subject_obj, latest_series in tasks:
                if not latest_series:
                    logger.error(f"Subject '{subject_name}' doesn't have any series")
                    continue

                if await latest_series.is_expired():
                    logger.info(
                        f"Subjects '{subject_name}' series {latest_series.id} is expired",
                    )
                    continue
                if await latest_series.is_closed():
                    logger.info(
                        f"Subjects '{subject_name}' series {latest_series.id} is closed",
                    )
                    continue

                logger.info(
                    f"Found relevant series {latest_series.id} for subject '{subject_name}'"
                )
                logger.info(
                    f"Adding subject '{subject_name}' into list of relevant subjects"
                )
                filtered_subjects.append(subject_obj)
        logger.info(f"Total relevant subjects found: {len(filtered_subjects)}")
        return filtered_subjects

    async def get_patch_by_id(self, id: int) -> dict:
        return await self.__get_object_by_id("patches", id)

    async def get_series(self, params: dict | None) -> list[Series]:
        return [
            Series(self, json)
            for json in await self.__get_objects_recursive("series", params=params)
        ]
