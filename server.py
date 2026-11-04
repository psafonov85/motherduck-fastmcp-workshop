"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import InputRequiredResult

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

mcp = FastMCP("sf-311")
log = logging.getLogger("sf311.server")
log.setLevel(logging.INFO)
if not log.handlers:
    # Without a handler, INFO lines never reach the deployment's server logs.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)

HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

CountGroup = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroup = Literal["neighborhood", "agency"]


def _check_name(value: str | None, choices: list[str], kind: str) -> None:
    """Raise a ToolError with "did you mean" suggestions for an unknown name."""
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(
        f"Unknown {kind} {value!r}.{hint} Call dataset_info for the exact {kind} names."
    )


def _mask_address(address: str) -> str:
    """Reduce a street address to its block: "994 HARRISON ST, ..." -> "900 block of HARRISON ST".

    Addresses without a leading house number, such as intersections, are returned unchanged.
    """
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """Return a request as a dict that is safe to show the model, with the address masked."""
    row = request.as_dict()
    row["address"] = _mask_address(request.address)
    return row


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name(service, SERVICES, "service")
    _check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: CountGroup | None = None,
) -> int | dict[str, int]:
    """Count 311 requests, optionally broken down by one field.

    Use this for any "how many", "which has the most", or "busiest" question.
    Without group_by it returns one number. With group_by it returns a dict of
    group -> count, largest first, at most 50 groups. group_by="weekday" groups
    by the day the request was opened (e.g. "Monday").

    service and neighborhood must be exact names from dataset_info. Use
    search_requests to see individual requests, and median_close_hours for how
    fast requests were closed.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def count_by_hour(
    service: str | None = None,
    neighborhood: str | None = None,
) -> dict[str, int]:
    """Count 311 requests by the hour of the day they were opened.

    Use this for "what time of day", "which hour", or "busiest hour"
    questions. Returns a dict of hour -> count, where the hour is "0" to "23"
    in San Francisco local time, busiest hour first. Hours with no matching
    requests are left out.

    service and neighborhood must be exact names from dataset_info. For counts
    by weekday, neighborhood, service, or anything other than hour, use
    count_requests.
    """
    counts = Counter(r.opened.hour for r in _matching(service, neighborhood))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {str(hour): count for hour, count in ranked}


OFFICIAL_BY_LOWER = {name.lower(): name for name in NEIGHBORHOODS}
LOCAL_BY_LOWER = {name.lower(): name for name in LOCAL_NAMES}
OVERLAPS: dict[str, list[str]] = {
    local: sorted({r.neighborhood for r in REQUESTS if r.local_name == local and r.neighborhood})
    for local in LOCAL_NAMES
}
NEIGHBORHOOD_CHOICE = "neighborhood_part"


@mcp.tool(annotations={"readOnlyHint": True})
def count_in_neighborhood(
    name: str,
    ctx: Context,
    boundary: str | None = None,
) -> int | str | InputRequiredResult:
    """Count 311 requests in a neighborhood, by official or local name.

    Use this when the user names a place, including local names such as
    "Lower Nob Hill" or "Duboce Triangle" that are not one of the 41 official
    neighborhoods. Returns one number.

    - An official neighborhood name counts that neighborhood.
    - A local name inside one official neighborhood counts that local area.
    - A local name that spans several official neighborhoods asks the user
      which part to count, unless `boundary` is given. Pass boundary="all" to
      count the whole local area, or one official neighborhood name to count
      only the part inside it. If the user declines, this returns a message
      and counts nothing.

    For breakdowns by service, weekday, or source, use count_requests with an
    official neighborhood name.
    """
    key = name.strip().lower()
    if key in OFFICIAL_BY_LOWER:
        official = OFFICIAL_BY_LOWER[key]
        return sum(1 for r in REQUESTS if r.neighborhood == official)

    if key not in LOCAL_BY_LOWER:
        suggestions = closest(name, NEIGHBORHOODS + LOCAL_NAMES)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(
            f"Unknown neighborhood {name!r}.{hint} "
            "Call dataset_info for the official neighborhood names."
        )

    local = LOCAL_BY_LOWER[key]
    in_area = [r for r in REQUESTS if r.local_name == local]
    parts = OVERLAPS[local]
    if len(parts) <= 1:
        return len(in_area)

    all_option = f"All of {local}"
    if boundary is None:
        choice = read_choice(ctx, NEIGHBORHOOD_CHOICE)
        if choice is None:
            return ask_choice(
                NEIGHBORHOOD_CHOICE,
                f"{local} spans {len(parts)} official neighborhoods. Which part should I count?",
                parts + [all_option],
            )
        if not choice.accepted:
            return (
                f"Nothing was counted: the user did not pick a part of {local}. "
                f"Call again with boundary set to one of {', '.join(parts)}, or 'all'."
            )
        boundary = choice.value

    if boundary.strip().lower() in ("all", all_option.lower()):
        return len(in_area)
    part = {p.lower(): p for p in parts}.get(boundary.strip().lower())
    if part is None:
        raise ToolError(
            f"{local} does not overlap {boundary!r}. "
            f"Use boundary='all' or one of: {', '.join(parts)}."
        )
    return sum(1 for r in in_area if r.neighborhood == part)


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: MedianGroup | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests.

    Use this for "how fast" or "how long does it take" questions. Open requests
    are ignored. Without group_by it returns one number rounded to 0.1 (or null
    if nothing matched is closed). With group_by it returns a dict of group ->
    median hours, fastest first, leaving out groups with fewer than 10 closed
    requests.

    service must be an exact name from dataset_info. Use count_requests for
    volumes and search_requests for individual requests.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {
        group: round(median(values), 1)
        for group, values in hours.items()
        if len(values) >= MIN_CLOSED_FOR_MEDIAN
    }
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests that match the filters, newest first.

    Addresses are shown only to the block (e.g. "900 block of HARRISON ST");
    exact house numbers are never available.

    Use this to show examples or details of requests, not to count them: it
    returns at most `limit` rows (1 to 50), so for totals, rankings, or
    "which day/neighborhood has the most" use count_requests instead.

    service and neighborhood must be exact names from dataset_info. Use
    get_request when you already have a request ID.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(
            f"limit must be between 1 and {MAX_ROWS}, got {limit}. "
            "To count all matching requests, use count_requests."
        )
    rows = _matching(service, neighborhood, status)
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return one 311 request by its ID, with all its fields.

    The address is shown only to the block (e.g. "900 block of HARRISON ST");
    exact house numbers are never available.

    Use this when you already have an ID, for example from search_requests.
    To find requests by service, neighborhood, or status, use search_requests.
    """
    log.info("get_request %s by %s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(
            f"No request with ID {request_id!r} in this week's data. "
            "Use search_requests to find request IDs."
        )
    return _public(request)


if __name__ == "__main__":
    mcp.run()
