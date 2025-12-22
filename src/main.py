"""
Main entry point for the Iran AQI IoT-ML pipeline.
This script orchestrates data collection, storage, and visualization.
"""

import logging

# from .config.settings import load_settings
# from .data_loader.aqi_api_client import AQIAPIClient
# from .pipeline.collector import collect_records
# from .storage.sqlite_storage import SQLiteStorage
# from .visualization.plots import PlotService

from .ml.train_uci_model import train_and_export_uci_model
from pathlib import Path
from .visualization.uci_plots import plot_actual_vs_predicted


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main() -> None:
    """
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

    # Train ML model on UCI dataset and export to ONNX
    uci_csv = Path("data/uci/AirQualityUCI.csv")
    onnx_out = Path("data/models/uci_co_model.onnx")

    train_and_export_uci_model(
        csv_path=uci_csv,
        out_path=onnx_out
    )
    """

    uci_csv = Path("data/uci/AirQualityUCI.csv")
    onnx_out = Path("data/models/uci_co_model.onnx")

    train_and_export_uci_model(
        csv_path=uci_csv,
        out_path=onnx_out
    )

    logging.info(f"ONNX model exported to: {onnx_out}")

    plot_path = Path("data/plots/uci_actual_vs_pred.png")
    plot_actual_vs_predicted(uci_csv, plot_path)

    logging.info(f"Visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()
