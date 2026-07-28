"""
services/export_service.py
Handles exporting subtitle/prediction history to TXT, CSV, and PDF.
"""

import csv
import io
from typing import List

from database.db_manager import HistoryRecord
from utils.logger import get_logger

logger = get_logger(__name__)


class ExportService:
    """Generates downloadable byte buffers for subtitle history exports."""

    def to_txt(self, records: List[HistoryRecord]) -> bytes:
        lines = [
            f"[{r.timestamp}] {r.sentence}  (gesture={r.gesture}, lip={r.lip_word}, "
            f"confidence={r.confidence:.2f})"
            for r in records
        ]
        return "\n".join(lines).encode("utf-8")

    def to_csv(self, records: List[HistoryRecord]) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["id", "timestamp", "gesture", "lip_word", "sentence", "confidence"])
        for r in records:
            writer.writerow([r.id, r.timestamp, r.gesture, r.lip_word, r.sentence, r.confidence])
        return buffer.getvalue().encode("utf-8")

    def to_pdf(self, records: List[HistoryRecord]) -> bytes:
        try:
            from fpdf import FPDF
        except ImportError:
            logger.error("fpdf2 not installed; cannot export PDF")
            raise

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "LipSync - Subtitle History", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(4)
        for r in records:
            text = f"[{r.timestamp}] {r.sentence} (gesture={r.gesture}, lip={r.lip_word}, conf={r.confidence:.2f})"
            pdf.multi_cell(0, 6, text)
        return bytes(pdf.output(dest="S"))
