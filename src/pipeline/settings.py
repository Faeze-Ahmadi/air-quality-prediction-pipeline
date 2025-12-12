from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Central project settings (loaded from environment)."""
    aqicn_api_token: str
    data_dir: Path
    db_path: Path
    cities: list[str]


def load_settings() -> Settings:
    load_dotenv()

    token = os.getenv("AQICN_API_TOKEN", "").strip()
    if not token or token == "YOUR_API_TOKEN_HERE":
        raise ValueError("AQICN_API_TOKEN is missing/invalid in .env")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    db_path = data_dir / "aqi_history.sqlite"

    # Phase-1 default cities (can expand later)
    cities = ["tehran", "isfahan", "mashhad", "ahvaz"]

    return Settings(
        aqicn_api_token=token,
        data_dir=data_dir,
        db_path=db_path,
        cities=cities,
    )
