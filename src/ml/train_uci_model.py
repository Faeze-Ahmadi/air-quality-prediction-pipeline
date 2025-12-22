import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def train_and_export_uci_model(csv_path: Path, out_path: Path) -> None:
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
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.4f}")

    initial_type = [("float_input", FloatTensorType([None, 1]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
