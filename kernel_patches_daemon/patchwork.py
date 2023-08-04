# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import re
from typing import Any, AnyStr, Callable, Dict, Final, List, Optional, Sequence, Set
from urllib.parse import urljoin

import dateutil.parser as dateparser
import requests
from cachetools import cached, TTLCache

from opentelemetry import metrics
from pyre_extensions import none_throws
from requests.adapters import HTTPAdapter

# when we want to push this patch through CI
RELEVANT_STATES: Dict[str, int] = {
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
IRRELEVANT_STATES: Dict[str, int] = {
    "rejected": 4,
    "accepted": 3,
    "not-applicable": 6,
    "superseded": 9,
    "awaiting-upstream": 8,
    "deferred": 10,
    "mainlined": 11,
    "handled-elsewhere": 17,
}

PW_CHECK_PENDING_STATES: Dict[Optional[str], str] = {
    None: "pending",
    "cancelled": "pending",
}

PW_CHECK_CONCLUSIVE_STATES: Dict[str, str] = {
    "success": "success",
    "skipped": "success",
    "warning": "warning",
    "failure": "fail",
}


PW_CHECK_STATES: Dict[Optional[str], str] = {
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


def log_response(func: Callable[..., requests.Response]):
    def log_response_content(response: requests.Response) -> None:
        log_level = logging.DEBUG if response.ok else logging.ERROR

        try:
            content = json_pprint(response.json())
        except json.decoder.JSONDecodeError:
            content = f"<binary content: {len(response.content)} bytes>"
        except Exception:
            logger.exception("Failed to handle Patchwork response")
            return

        logger.log(
            log_level,
            f"Patchwork {response.request.method} {response.request.url} "
            f"{response.status_code}, response: {content}",
        )

    def log_response_metrics(response: requests.Response) -> None:
        response_code_leading_num = response.status_code // 100

        if not response.request.method:
            logger.error("No http method founds in response, skipping metrics")
            return

        normalized_http_method = response.request.method.strip().lower()
        api_requests.add(
            1,
            {
                "http_code": f"{response_code_leading_num}xx",
                "http_method": normalized_http_method,
            },
        )
        api_requests_time.record(
            response.elapsed.microseconds // 1000,
            {"http_method": normalized_http_method},
        )

    def wrapper(*args, **kwargs) -> requests.Response:
        response = func(*args, **kwargs)

        log_response_content(response)
        log_response_metrics(response)

        return response

    return wrapper


def json_pprint(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True)


def time_since_secs(date: str) -> float:
    parsed_datetime = dateparser.parse(date)
    duration = datetime.datetime.utcnow() - parsed_datetime
    return duration.total_seconds()


def parse_tags(input: str) -> Set[str]:
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


class Subject:
    def __init__(self, subject: str, pw_client: "Patchwork") -> None:
        self.pw_client = pw_client
        self.subject = subject

    @property
    def branch(self) -> Optional[str]:
        if len(self.relevant_series) == 0:
            return None
        return f"series/{self.relevant_series[0].id}"

    @property
    def latest_series(self) -> Optional["Series"]:
        if len(self.relevant_series) == 0:
            return None
        return self.relevant_series[-1]

    @property
    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def relevant_series(self) -> List["Series"]:
        """
        cache and return sorted list of relevant series
        where first element is first known version of same subject
        and last is the most recent
        """
        series_list = self.pw_client.get_series(params={"q": self.subject})
        logging.debug(
            f"All series for '{self.subject}' subject: {json_pprint([s.to_json() for s in series_list])}"
        )
        # we using full text search which could give ambiguous results
        # so we must filter out irrelevant results
        relevant_series = [
            series
            for series in series_list
            if series.subject == self.subject and series.has_matching_patches()
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
    def __init__(self, pw_client: "Patchwork", data: Dict) -> None:
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
        self.mbox = data["mbox"]
        self.patches = data.get("patches", [])
        self.cover_letter = data.get("cover_letter")

        try:
            logging.debug(f"Parsing suject name from '{self.name}' series name")
            self.subject = none_throws(re.match(SUBJECT_REGEXP, data["name"])).group(
                "name"
            )
        except Exception as ex:
            raise ValueError(
                f"Failed to parse subject from series name '{data['name']}'"
            ) from ex

    def _is_patch_matching(self, patch: Dict[str, Any]) -> bool:
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
    def get_patches(self) -> List[Dict]:
        """
        Returns patches preserving original order
        for the most recent relevant series
        """
        return [self.pw_client.get_patch_by_id(patch["id"]) for patch in self.patches]

    def is_closed(self) -> bool:
        """
        Series considered closed if at least one patch in this series
        is in irrelevant states
        """
        for patch in self.get_patches():
            if patch["state"] in IRRELEVANT_STATES:
                return True
        return False

    @cached(cache=TTLCache(maxsize=1, ttl=120))
    def all_tags(self) -> Set[str]:
        """
        Tags fetched from series name, diffs and cover letter
        for most relevant series
        """
        tags = {f"V{self.version}"}

        for patch in self.get_patches():
            tags |= parse_tags(patch["name"])
            tags.add(patch["state"])

        if self.cover_letter:
            tags |= parse_tags(self.cover_letter["name"])

        tags |= parse_tags(self.name)

        return tags

    def visible_tags(self) -> Set[str]:
        return {f"V{self.version}", *[diff["state"] for diff in self.get_patches()]}

    def is_expired(self) -> bool:
        for diff in self.get_patches():
            if diff["state"] in TTL:
                if time_since_secs(diff["date"]) >= TTL[diff["state"]]:
                    return True
        return False

    @cached(cache=TTLCache(maxsize=1, ttl=120))
    def get_patch_binary_content(self) -> bytes:
        content = self.pw_client.get_blob(self.mbox)
        logger.debug(
            f"Received patch mbox for series {self.id}, size: {len(content)} bytes"
        )
        return content

    def has_matching_patches(self) -> bool:
        for patch in self.get_patches():
            if self._is_patch_matching(patch):
                return True

        return False

    def set_check(self, **kwargs) -> None:
        for patch in self.get_patches():
            self.pw_client.post_check_for_patch_id(
                patch_id=patch["id"], check_data=kwargs
            )

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


class Patchwork:
    def __init__(
        self,
        server: str,
        search_patterns: List[Dict[str, Any]],
        auth_token: Optional[str] = None,
        lookback_in_days: int = 7,
        api_version: str = "1.2",
        http_retries: Optional[int] = None,
    ) -> None:
        self.api_url = f"https://{server}/api/{api_version}/"
        self.auth_token = auth_token
        if not auth_token:
            logger.warning("Patchwork client runs in read-only mode")
        self.search_patterns = search_patterns
        self.since = self.format_since(lookback_in_days)
        # member variable initializations
        self.known_series: Dict[int, Series] = {}
        self.known_subjects: Dict[str, Subject] = {}
        self.http_session = requests.Session()
        adapter = HTTPAdapter(max_retries=http_retries)

        self.http_session.mount("http://", adapter)
        self.http_session.mount("https://", adapter)

    def format_since(self, pw_lookback: int) -> str:
        today = datetime.datetime.utcnow().date()
        lookback = today - datetime.timedelta(days=pw_lookback)
        return lookback.strftime("%Y-%m-%dT%H:%M:%S")

    @log_response
    def __get(self, path: AnyStr, **kwargs: Dict) -> requests.Response:
        # pyre-ignore
        return self.http_session.get(url=urljoin(self.api_url, path), **kwargs)

    def __get_object_by_id(self, object_type: str, object_id: int) -> Dict:
        return self.__get(f"{object_type}/{object_id}/").json()

    def __get_objects_recursive(
        self, object_type: str, params: Optional[Dict] = None
    ) -> List[Dict]:
        items = []
        path = f"{object_type}/"
        while True:
            response = self.__get(path, params=params)
            items += response.json()

            if "next" not in response.links:
                break

            path = response.links["next"]["url"]
        return items

    @log_response
    def __post(self, path: AnyStr, data: Dict) -> requests.Response:
        return self.http_session.post(
            # pyre-ignore
            url=urljoin(self.api_url, path),
            headers={"Authorization": f"Token {self.auth_token}"},
            data=data,
        )

    def __try_post(self, path: AnyStr, data: Dict) -> Optional[requests.Response]:
        if not self.auth_token:
            logger.warning(f"Patchwork POST {path}: read-only mode, request ignored")
            logger.debug(f"Patchwork POST data: {json_pprint(data)}")
            return None

        return self.__post(path, data)

    def get_blob(self, url: AnyStr) -> bytes:
        return self.__get(url, allow_redirects=True).content

    def get_latest_check_for_patch(
        self,
        patch_id: int,
        context: str,
    ) -> Dict[str, Any]:
        """
        Return dict of the latest check with matching context and patch_id if exists.
        Returns None if such check does not exist.
        """
        # get single latest one with matching context value
        # patchwork.kernel.org api ignores case for context query
        checks = self.__get(
            f"patches/{patch_id}/checks/",
            params={
                "context": context,
                "order": "-date",
                "per_page": "1",
            },
        ).json()
        logger.debug(
            f"Received latest check for patch {patch_id} with context {context}: {json_pprint(checks)}"
        )

        if len(checks) == 0:
            raise ValueError(
                f"Patch {patch_id} doesn't have any checks with '{context}' context"
            )

        return checks[0]

    def post_check_for_patch_id(
        self, patch_id: int, check_data: Dict[str, Any]
    ) -> Optional[requests.Response]:
        new_state = PW_CHECK_STATES.get(check_data["state"], PW_CHECK_STATES[None])
        updated_check_data = {
            **check_data,
            "state": new_state,
        }
        logger.debug(
            f"Trying to update check for {patch_id} with a new content: {json_pprint(updated_check_data)}"
        )
        try:
            check = self.get_latest_check_for_patch(patch_id, check_data["context"])
            if (
                check_data["state"] in PW_CHECK_PENDING_STATES
                and PW_CHECK_PENDING_STATES[check_data["state"]] != check["state"]
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
                logger.info(
                    f"Not posting state update for patch {patch_id}: previous state '{new_state}' and url are the same"
                )
                return None

            logger.info(
                f"Updating patch {patch_id} check, current state: '{check.get('state')}', new state: '{new_state}'"
            )
        except ValueError:
            logger.info(f"Setting patch {patch_id} check to '{new_state}' state")

        return self.__try_post(
            f"patches/{patch_id}/checks/",
            data=updated_check_data,
        )

    def get_series_by_id(self, series_id: int) -> Series:
        # fetches directly only if series is not available in local scope
        if series_id not in self.known_series:
            self.known_series[series_id] = Series(
                self, self.__get_object_by_id("series", series_id)
            )

        return self.known_series[series_id]

    def get_subject_by_series(self, series: Series) -> Subject:
        # local cache for subjects
        if series.subject not in self.known_subjects:
            subject = Subject(series.subject, self)
            self.known_subjects[series.subject] = subject

        return self.known_subjects[series.subject]

    def get_relevant_subjects(self) -> Sequence[Subject]:
        subjects = {}
        filtered_subjects = []
        self.known_series = {}
        self.known_subjects = {}

        for pattern in self.search_patterns:
            patch_filters = {
                "since": self.since,
                "state": RELEVANT_STATES.values(),
                "archived": False,
            }
            patch_filters.update(pattern)
            logger.info(
                f"Searching for Patchwork patches that match the criteria: {patch_filters}"
            )
            all_patches = self.__get_objects_recursive("patches", params=patch_filters)
            for patch in all_patches:
                patch_series = patch["series"]
                for series_data in patch_series:
                    if series_data["name"]:
                        series = Series(self, series_data)
                        logger.debug(f"Adding {series.id} into list of known series.")
                        self.known_series[series.id] = series
                    else:
                        err_malformed_series.add(1)
                        logger.error(f"Malformed series: {series_data}")
                        continue

                    if series.subject not in subjects:
                        subjects[series.subject] = Subject(series.subject, self)
                        self.known_subjects[series.subject] = subjects[series.subject]

            logger.info(f"Total subjects found: {len(subjects)}")
            for subject_name, subject_obj in subjects.items():
                latest_series = subject_obj.latest_series
                if not latest_series:
                    logger.error(f"Subject '{subject_name}' doesn't have any series")
                    continue

                if latest_series.is_expired():
                    logger.info(
                        f"Subjects '{subject_name}' series {latest_series.id} is expired",
                    )
                    continue
                if latest_series.is_closed():
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

    def get_patch_by_id(self, id: int) -> Dict:
        return self.__get_object_by_id("patches", id)

    def get_series(self, params: Optional[Dict]) -> List[Series]:
        return [
            Series(self, json)
            for json in self.__get_objects_recursive("series", params=params)
        ]
