from pathlib import Path
from src.ml.train_aqicn_model import train_and_export_aqicn_model

if __name__ == "__main__":
    mae = train_and_export_aqicn_model(
        db_path=Path("data/aqi_history.sqlite"),
        onnx_out=Path("data/models/aqicn_aqi_model.onnx"),
    )

    print("Training finished.")
    print("Final MAE:", mae)
