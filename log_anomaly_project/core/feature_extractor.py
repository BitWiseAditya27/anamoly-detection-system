"""
feature_extractor.py
Converts parsed log records into a numeric feature matrix.

Features (per log line):
  1. level_score        - INFO=0, WARNING=1, ERROR=2, CRITICAL=3, UNKNOWN=2 (treat as risky)
  2. message_length      - length of the message text
  3. has_number          - 1 if message contains a digit (error codes, IPs, counts)
  4. keyword_score       - count of suspicious keywords present (failure, unauthorized, blocked, etc.)
  5. template_frequency  - how common this "normalized" message template is in the batch
                           (rare templates are more suspicious)
  6. events_per_minute   - how many log lines occurred in the same minute (burst detection,
                           useful for brute-force / flood style anomalies)
  7. is_offhours         - 1 if the timestamp falls outside typical business hours (0-6, 22-23)

Frequency-style features (#5, #6) are computed relative to the batch being analyzed,
which is what lets the model catch "rare in THIS log file" patterns rather than
relying on a fixed, hand-coded blacklist.
"""

import re
from collections import Counter
import numpy as np

SUSPICIOUS_KEYWORDS = [
    "failure", "fail", "exception", "unauthorized", "denied", "blocked",
    "timeout", "suspicious", "injection", "brute force", "privilege",
    "malformed", "unknown source", "rate limit", "escalation",
]

LEVEL_SCORE = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3, "UNKNOWN": 2, "DEBUG": 0}

NUMBER_RE = re.compile(r"\d+")


def normalize_template(message):
    """Collapse variable parts (numbers, IPs) so similar messages map to the same template.
    e.g. 'Rate limit exceeded for user_482' -> 'Rate limit exceeded for user_<NUM>'
    """
    return NUMBER_RE.sub("<NUM>", message)


def keyword_score(message):
    msg_lower = message.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in msg_lower)


def extract_features(records):
    """
    records: list of dicts from parser.py (must have 'message', 'level', 'timestamp')
    Returns: (feature_matrix: np.ndarray, feature_names: list[str], extra: dict of per-line metadata)
    """
    templates = [normalize_template(r["message"]) for r in records]
    template_counts = Counter(templates)
    n = len(records)

    # events per minute (bucket by truncated-to-minute timestamp)
    minute_buckets = Counter()
    for r in records:
        if r["timestamp"] is not None:
            bucket = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        else:
            bucket = "unknown"
        minute_buckets[bucket] += 1

    feature_rows = []
    per_line_meta = []

    for r, tmpl in zip(records, templates):
        level = r["level"]
        message = r["message"]
        ts = r["timestamp"]

        f_level = LEVEL_SCORE.get(level, 2)
        f_len = len(message)
        f_has_num = 1 if NUMBER_RE.search(message) else 0
        f_keyword = keyword_score(message)
        f_template_freq = template_counts[tmpl] / n  # fraction of batch sharing this template
        if ts is not None:
            bucket = ts.strftime("%Y-%m-%d %H:%M")
            f_events_per_min = minute_buckets[bucket]
            f_offhours = 1 if (ts.hour < 6 or ts.hour >= 22) else 0
        else:
            f_events_per_min = 1
            f_offhours = 0

        feature_rows.append([
            f_level, f_len, f_has_num, f_keyword,
            f_template_freq, f_events_per_min, f_offhours,
        ])
        per_line_meta.append({"template": tmpl})

    feature_names = [
        "level_score", "message_length", "has_number", "keyword_score",
        "template_frequency", "events_per_minute", "is_offhours",
    ]

    X = np.array(feature_rows, dtype=float)
    return X, feature_names, per_line_meta
