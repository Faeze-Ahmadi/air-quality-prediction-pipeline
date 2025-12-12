from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.storage.sqlite_storage import AQIRecord


@dataclass
class CollectorResult:
    records: list[AQIRecord]
    errors: list[str]


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        # Some APIs can return "-" or strings
        return float(x)
    except Exception:
        return None


def collect_records(client, cities: list[str]) -> CollectorResult:
    """
    Collect one snapshot for multiple cities.
    `client` is expected to have: fetch_city_aqi(city)->dict
    """
    records: list[AQIRecord] = []
    errors: list[str] = []

    for city in cities:
        try:
            d = client.fetch_city_aqi(city)
            records.append(
                AQIRecord(
                    city=city,
                    aqi=_to_float(d.get("aqi")),
                    pm25=_to_float(d.get("pm25")),
                    pm10=_to_float(d.get("pm10")),
                    co=_to_float(d.get("co")),
                    no2=_to_float(d.get("no2")),
                    so2=_to_float(d.get("so2")),
                    o3=_to_float(d.get("o3")),
                    timestamp=d.get("timestamp") or datetime.utcnow().isoformat(),
                )
            )
        except Exception as e:
            errors.append(f"{city}: {e}")

    return CollectorResult(records=records, errors=errors)
