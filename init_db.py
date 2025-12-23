from pathlib import Path
from src.storage.sqlite_storage import SQLiteStorage

if __name__ == "__main__":
    db_path = Path("data/aqi_history.sqlite")
    SQLiteStorage(db_path)
    print("Database initialized and table created.")
