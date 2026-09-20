"""Shared deterministic fixtures: a fixed 'now' inside US market hours and a
history whose points all fall in the same (market-open) regime, so scoring
never depends on the wall clock the tests happen to run at."""

from datetime import datetime, timedelta, timezone

# Tuesday 2026-09-15 15:00 UTC = 11:00 EDT -> market open
FIXED_NOW = datetime(2026, 9, 15, 15, 0, tzinfo=timezone.utc)


class FixedDatetime(datetime):
    """Drop-in for main.datetime whose now() is FIXED_NOW."""
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


def open_market_stamps(n: int, end: datetime = FIXED_NOW) -> list[str]:
    """n ISO timestamps, all inside regular US market hours, three per
    business day (14:00/16:00/18:00 UTC = 10:00/12:00/14:00 EDT) ending
    shortly before `end`. Points within a day are 2h apart (valid for the
    movement component); weekends and the 2026-09-07 holiday are skipped."""
    stamps, day = [], end.replace(hour=0, minute=0, second=0, microsecond=0)
    while len(stamps) < n:
        if day.weekday() < 5 and day.date().isoformat() != "2026-09-07":
            for hour in (18, 16, 14):
                stamp = day.replace(hour=hour)
                if stamp < end and len(stamps) < n:
                    stamps.append(stamp.isoformat())
        day -= timedelta(days=1)
    return list(reversed(stamps))
