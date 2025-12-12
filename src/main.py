import logging

from .config.settings import load_settings
from .data_loader.aqi_api_client import AQIAPIClient
from .pipeline.collector import collect_records
from .storage.sqlite_storage import SQLiteStorage
from .visualization.plots import PlotService



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main() -> None:
    settings = load_settings()

    client = AQIAPIClient()  # uses .env internally too (OK for now)
    storage = SQLiteStorage(settings.db_path)

    result = collect_records(client, settings.cities)

    if result.errors:
        for e in result.errors:
            logging.warning(f"Collector error: {e}")

    inserted = storage.insert_many(result.records)
    logging.info(f"Inserted {inserted} rows into SQLite DB: {settings.db_path}")

    latest = storage.fetch_latest_per_city()

    plotter = PlotService(settings.data_dir / "plots")
    out = plotter.plot_latest_aqi_bar(latest, filename="latest_aqi.png")
    logging.info(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
