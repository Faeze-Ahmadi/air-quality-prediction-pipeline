from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


FEATURE_COLS = ["pm25", "pm10", "co", "no2", "so2", "o3"]
TARGET_COL = "aqi"


def load_aqicn_dataframe(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM aqi_readings", conn)
    conn.close()
    return df


def preprocess_aqicn(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    for col in FEATURE_COLS + [TARGET_COL]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return work


def train_and_export_aqicn_model(
    db_path: Path,
    onnx_out: Path,
) -> float:
    df = load_aqicn_dataframe(db_path)
    df = preprocess_aqicn(df)

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Export ONNX
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    onnx_out.write_bytes(onnx_model.SerializeToString())

    # Test ONNXRuntime
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run(None, {input_name: X_test.to_numpy().astype(np.float32)})[0]

    mae_onnx = mean_absolute_error(y_test.to_numpy(), onnx_preds)

    print(f"AQICN MAE (sklearn): {mae:.2f}")
    print(f"AQICN MAE (onnxruntime): {mae_onnx:.2f}")

    return mae_onnx
