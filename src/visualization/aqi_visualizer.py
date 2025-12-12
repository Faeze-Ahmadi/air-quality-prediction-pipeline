from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class AQIVisualizer:
    """
    Visualization utilities for AQI historical data.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def plot_latest_aqi(self, filename: str) -> None:
        file_path = self.data_dir / filename
        df = pd.read_csv(file_path)

        # Convert AQI to numeric, force invalid values to NaN
        df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")

        # Drop rows with invalid AQI
        df = df.dropna(subset=["aqi"])

        # Select latest VALID AQI per city
        latest = (
            df.sort_values("timestamp")
            .groupby("city", as_index=False)
            .tail(1)
        )

        plt.figure(figsize=(8, 5))
        bars = plt.bar(latest["city"], latest["aqi"])

        plt.title("Latest Valid AQI by City", fontsize=14)
        plt.xlabel("City")
        plt.ylabel("AQI")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        plt.show()

