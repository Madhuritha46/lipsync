"""
tests/test_db.py
Unit tests for database/db_manager.py using a temporary SQLite file so
the real project database is never touched by tests.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DBManager  # noqa: E402


def _temp_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let DBManager create it fresh
    return path


def test_insert_and_fetch():
    db = DBManager(db_path=_temp_db_path())
    db.insert_prediction("STOP", "HELP", "STOP AND HELP ME", 0.87)
    records = db.fetch_history()
    assert len(records) == 1
    assert records[0].sentence == "STOP AND HELP ME"
    assert abs(records[0].confidence - 0.87) < 1e-6


def test_clear_history():
    db = DBManager(db_path=_temp_db_path())
    db.insert_prediction("OK", "YES", "CONFIRMED, PROCEED", 0.9)
    db.clear_history()
    assert db.fetch_history() == []


def test_fetch_limit_and_order():
    db = DBManager(db_path=_temp_db_path())
    for i in range(5):
        db.insert_prediction("OK", "YES", f"SENTENCE {i}", 0.5)
    records = db.fetch_history(limit=2)
    assert len(records) == 2
    # Most recent first
    assert records[0].sentence == "SENTENCE 4"


if __name__ == "__main__":
    test_insert_and_fetch()
    test_clear_history()
    test_fetch_limit_and_order()
    print("All db_manager tests passed.")
