"""
database/db_manager.py
SQLite-backed persistence layer for prediction history.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from config import DB_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HistoryRecord:
    id: int
    timestamp: str
    gesture: str
    lip_word: str
    sentence: str
    confidence: float


class DBManager:
    """Thin wrapper around sqlite3 for storing/retrieving prediction history."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    gesture TEXT,
                    lip_word TEXT,
                    sentence TEXT,
                    confidence REAL
                )
                """
            )
            conn.commit()
        logger.info("Database ready at %s", self._db_path)

    def insert_prediction(self, gesture: str, lip_word: str, sentence: str,
                           confidence: float) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO history (timestamp, gesture, lip_word, sentence, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, gesture, lip_word, sentence, confidence),
            )
            conn.commit()

    def fetch_history(self, limit: int = 200) -> List[HistoryRecord]:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, gesture, lip_word, sentence, confidence "
                "FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [HistoryRecord(*row) for row in rows]

    def clear_history(self) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM history")
            conn.commit()
        logger.info("History cleared")
