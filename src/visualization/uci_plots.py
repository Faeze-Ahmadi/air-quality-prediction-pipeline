import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def plot_actual_vs_predicted(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    df["CO(GT)"] = pd.to_numeric(df["CO(GT)"], errors="coerce")
    df["PT08.S1(CO)"] = pd.to_numeric(df["PT08.S1(CO)"], errors="coerce")

    df = df.dropna(subset=["CO(GT)", "PT08.S1(CO)"])

    X = df[["PT08.S1(CO)"]]
    y = df["CO(GT)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual CO(GT)")
    plt.ylabel("Predicted CO(GT)")
    plt.title("UCI Air Quality: Actual vs Predicted")
    plt.grid(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
