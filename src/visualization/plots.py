from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


class PlotService:
    """Creates presentation-ready plots."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True)

    def plot_latest_aqi_bar(self, latest_rows: list[dict], filename: str = "latest_aqi.png") -> Path:
        df = pd.DataFrame(latest_rows)

        # Ensure numeric
        df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
        df = df.dropna(subset=["aqi"]).sort_values("aqi")

        plt.figure(figsize=(9, 5))
        bars = plt.bar(df["city"], df["aqi"])
        plt.title("Latest Valid AQI by City")
        plt.xlabel("City")
        plt.ylabel("AQI")

        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom")

        out_path = self.out_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
