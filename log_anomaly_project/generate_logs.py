"""
generate_logs.py
Generates a synthetic log dataset with:
  - data/train_logs.csv   -> logs + hidden ground-truth label (for evaluation ONLY, not fed to the model)
  - data/sample_upload.log -> a plain, unlabeled log file to simulate what a user would upload

Log format: "YYYY-MM-DD HH:MM:SS LEVEL message"
"""

import random
from datetime import datetime, timedelta
import csv

random.seed(42)

NORMAL_MESSAGES = [
    ("INFO", "API request received: GET /products"),
    ("INFO", "API request received: GET /users"),
    ("INFO", "API request received: POST /orders"),
    ("INFO", "User login successful for user_{n}"),
    ("INFO", "Database connection established"),
    ("INFO", "Cache refreshed successfully"),
    ("INFO", "Health check passed"),
    ("WARNING", "Slow query detected: {n}ms"),
    ("WARNING", "Database connection retried"),
    ("WARNING", "Cache miss for key user_{n}"),
    ("ERROR", "Database connection failed"),          # occurs often enough to look "normal"
    ("ERROR", "Payment gateway timeout"),
]

ANOMALY_MESSAGES = [
    ("ERROR", "Suspicious IP access blocked from 192.168.{n}.{n2}"),
    ("ERROR", "Rate limit exceeded for user_{n}"),
    ("ERROR", "Unauthorized access attempt detected"),
    ("ERROR", "Multiple failed login attempts for user_{n}"),
    ("WARNING", "Unusual outbound traffic detected: {n}MB"),
    ("ERROR", "SQL injection pattern detected in request"),
    ("ERROR", "Privilege escalation attempt blocked"),
    ("INFO", "Config file accessed outside business hours"),  # anomaly despite being INFO
    ("ERROR", "Malformed request payload from unknown source"),
    ("ERROR", "Brute force login pattern detected for user_{n}"),
]


def fill(template):
    return template.format(n=random.randint(1, 999), n2=random.randint(1, 255))


def _random_business_hour_timestamp(day_start):
    """Pick a random time within a day, weighted toward business hours (9am-9pm),
    with a smaller tail of night-time traffic - realistic for both train and upload sets."""
    if random.random() < 0.85:
        hour = random.randint(9, 20)  # daytime traffic (majority)
    else:
        hour = random.choice(list(range(0, 9)) + list(range(21, 24)))  # off-hours (minority)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return day_start.replace(hour=hour, minute=minute, second=second)


def generate_dataset(n_normal=1800, n_anomaly=200, start_time=None, n_days=4):
    """Returns list of (timestamp, level, message, label) sorted by time, label=1 normal, -1 anomaly (sklearn convention).
    Spreads events across n_days with a realistic business-hours-weighted time-of-day distribution,
    so the 'off-hours' feature has a consistent, comparable distribution across any dataset generated
    with this function (training set, upload sample, etc.) regardless of total event count."""
    if start_time is None:
        start_time = datetime(2026, 7, 1)

    rows = []
    for _ in range(n_normal):
        day = start_time + timedelta(days=random.randint(0, n_days - 1))
        t = _random_business_hour_timestamp(day)
        level, template = random.choice(NORMAL_MESSAGES)
        rows.append([t, level, fill(template), 1])

    for _ in range(n_anomaly):
        day = start_time + timedelta(days=random.randint(0, n_days - 1))
        # anomalies skew slightly more toward off-hours (realistic for attacks), but not exclusively
        if random.random() < 0.4:
            hour = random.choice(list(range(0, 9)) + list(range(21, 24)))
            t2 = day.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
        else:
            t2 = _random_business_hour_timestamp(day)
        level, template = random.choice(ANOMALY_MESSAGES)
        rows.append([t2, level, fill(template), -1])

    rows.sort(key=lambda r: r[0])
    return rows


def write_labeled_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "level", "message", "label"])
        for t, level, msg, label in rows:
            writer.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), level, msg, label])


def write_plain_log(rows, path):
    with open(path, "w") as f:
        for t, level, msg, _ in rows:
            f.write(f"{t.strftime('%Y-%m-%d %H:%M:%S')} {level} {msg}\n")


if __name__ == "__main__":
    all_rows = generate_dataset(n_normal=1800, n_anomaly=200)
    write_labeled_csv(all_rows, "data/train_logs.csv")

    # A separate smaller, unlabeled file to simulate a real user upload
    upload_rows = generate_dataset(n_normal=300, n_anomaly=25,
                                    start_time=datetime(2026, 7, 20), n_days=2)
    write_plain_log(upload_rows, "data/sample_upload.log")

    print(f"Generated {len(all_rows)} labeled rows -> data/train_logs.csv")
    print(f"Generated {len(upload_rows)} unlabeled rows -> data/sample_upload.log")
