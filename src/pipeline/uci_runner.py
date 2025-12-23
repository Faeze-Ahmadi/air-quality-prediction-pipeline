from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.data_loader.uci_loader import load_uci_air_quality, preprocess_uci_for_co_regression
from src.visualization.uci_plots import (
    plot_actual_vs_predicted,
    plot_error_histogram,
)


logger = logging.getLogger(__name__)


def run_uci_pipeline(
    uci_csv: Path,
    onnx_out: Path,
    plot_out: Path,
) -> None:
    """
    UCI Core pipeline required by the course:
    Load -> Preprocess -> Train -> Evaluate -> Export ONNX -> Load ONNX (onnxruntime) -> Predict -> Plot
    """
    if not uci_csv.exists():
        raise FileNotFoundError(f"UCI dataset not found: {uci_csv}")

    # 1) Load + preprocess
    raw = load_uci_air_quality(uci_csv)
    df = preprocess_uci_for_co_regression(raw)

    # 2) Features/target (simple baseline, explainable)
    X = df[["PT08.S1(CO)"]].astype(float)
    y = df["CO(GT)"].astype(float)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Train sklearn model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5) Evaluate sklearn
    sk_preds = model.predict(X_test)
    mae_sklearn = mean_absolute_error(y_test, sk_preds)
    logger.info("UCI MAE (sklearn): %.4f", mae_sklearn)

    # 6) Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    onnx_out.write_bytes(onnx_model.SerializeToString())
    logger.info("ONNX exported: %s", onnx_out)

    # 7) Load & Predict using onnxruntime (explicit course requirement)
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    onnx_preds = sess.run(
        None, {input_name: X_test.to_numpy().astype(np.float32)}
    )[0].ravel()

    mae_onnx = mean_absolute_error(y_test.to_numpy(), onnx_preds)
    logger.info("UCI MAE (onnxruntime): %.4f", mae_onnx)

    # 8) Visualization (Actual vs Predicted)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    # Visualization 1: Actual vs Predicted (with MAE and y=x line)
    plot_actual_vs_predicted(
        y_true=y_test.to_numpy(),
        y_pred=onnx_preds,
        out_path=plot_out,
        mae=mae_onnx,
        title="UCI Air Quality: Actual vs Predicted (ONNXRuntime)",
    )

    # Visualization 2: Prediction error histogram
    error_plot_path = plot_out.parent / "uci_prediction_error_hist.png"
    plot_error_histogram(
        y_true=y_test.to_numpy(),
        y_pred=onnx_preds,
        out_path=error_plot_path,
    )

    logger.info("Visualization saved: %s", plot_out)
    logger.info("Error histogram saved: %s", error_plot_path)

