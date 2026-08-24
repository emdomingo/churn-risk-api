"""Structured JSON logging to stdout, for CloudWatch.

Lambda captures whatever the process writes to stdout and ships each line to CloudWatch
Logs, so there is no handler to install and no agent to run -- one line of JSON per event
is the whole integration. CloudWatch Logs Insights parses those lines natively, which is
what makes `stats avg(duration_ms) by path` possible without a log-parsing rule.

**Redaction is structural, not procedural** (`SPEC.md` §6: "request id, latency, score --
not raw PII"). The formatter emits only the keys in `LOGGABLE_FIELDS`; anything else
attached to a log call is dropped on the floor. A future caller who writes
`logger.info("scored", extra={"customerID": ...})` therefore leaks nothing -- the field
simply never reaches the output. An allowlist that must be *remembered* at each call site
is the kind that eventually fails, so this one is enforced in the only place every record
passes through.

The one thing this cannot protect is the message string itself: `logger.info(f"scored
{customer_id}")` would still print. Log events here are fixed literals with structured
fields alongside, never interpolated records, for exactly that reason.
"""

import json
import logging
import sys
from typing import Any

# The complete set of structured fields that may appear in a log line. Adding to this
# list is a decision about what leaves the process, so it should be a deliberate edit.
LOGGABLE_FIELDS = frozenset(
    {
        "request_id",  # correlates a line with the caller's request, and with API Gateway
        "method",
        "path",
        "status",
        "duration_ms",  # latency, the SPEC's second named field
        "batch_size",  # how many records, never which ones
        "model_version",
        "band",  # low/medium/high -- an aggregate, not an attribute of a person
        "probability",  # the SPEC's third named field
        "error",  # exception class name, never its message: those quote input
    }
)

# `logging.LogRecord` sets these itself; they are not caller-supplied extras and must not
# be mistaken for them.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with a fixed core and an allowlisted tail."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in LOGGABLE_FIELDS and key not in _RESERVED:
                payload[key] = value

        if record.exc_info:
            # The class name only. A traceback or an exception message can quote the
            # input that caused it, which is how request bodies end up in logs.
            payload["error"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Point the root logger at stdout with the JSON formatter. Idempotent.

    Replaces existing handlers rather than adding to them: both Lambda and uvicorn
    install their own, and leaving those in place gives every event twice -- once as
    JSON and once as unparseable text.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn's access log duplicates the request middleware's line, in its own format.
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn.access").handlers = []
