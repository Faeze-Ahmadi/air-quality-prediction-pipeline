import pandas as pd
from pathlib import Path


def load_uci_air_quality(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    # drop last empty column if exists
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]

    # parse datetime
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )

    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    return df
