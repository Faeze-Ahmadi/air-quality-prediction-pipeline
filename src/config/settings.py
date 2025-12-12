from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass
class Settings:
    aqicn_api_token: str
    data_dir: Path
    db_path: Path
    cities: list[str]


def load_settings() -> Settings:
    """
    Load project settings from environment variables.
    """
    load_dotenv()

    token = os.getenv("AQICN_API_TOKEN")
    if not token:
        raise RuntimeError(
            "AQICN_API_TOKEN not found. Make sure it exists in .env file."
        )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    db_path = data_dir / "aqi_history.sqlite"

    cities = ["tehran", "isfahan", "mashhad", "ahvaz"]

    return Settings(
        aqicn_api_token=token,
        data_dir=data_dir,
        db_path=db_path,
        cities=cities,
    )
