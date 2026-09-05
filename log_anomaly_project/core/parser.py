"""
parser.py
Parses raw log lines of the form:
    "YYYY-MM-DD HH:MM:SS LEVEL message text"
into structured dicts. Falls back gracefully on lines that don't match.
"""

import re
from datetime import datetime

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>INFO|WARNING|ERROR|DEBUG|CRITICAL)\s+"
    r"(?P<message>.*)$"
)


def parse_line(line, line_no=None):
    line = line.strip()
    if not line:
        return None

    match = LOG_PATTERN.match(line)
    if match:
        ts_str = match.group("timestamp")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts = None
        return {
            "line_no": line_no,
            "timestamp": ts,
            "timestamp_raw": ts_str,
            "level": match.group("level"),
            "message": match.group("message"),
            "raw": line,
        }

    # Unrecognized format: still keep the line so it isn't silently dropped.
    # Malformed lines are themselves often worth flagging.
    return {
        "line_no": line_no,
        "timestamp": None,
        "timestamp_raw": None,
        "level": "UNKNOWN",
        "message": line,
        "raw": line,
    }


def parse_log_text(text):
    """Parses a full log file's text content into a list of structured records."""
    records = []
    for i, line in enumerate(text.splitlines(), start=1):
        rec = parse_line(line, line_no=i)
        if rec is not None:
            records.append(rec)
    return records


def parse_log_file(path):
    with open(path, "r", errors="ignore") as f:
        return parse_log_text(f.read())
