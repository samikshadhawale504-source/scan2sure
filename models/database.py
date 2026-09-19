"""
SQLite database storage for Legal Metrology inspections.
Stores inspection audit logs, extracted fields, compliance reports, and evidence.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "inspections.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                product_name TEXT,
                image_filename TEXT,
                engine_used TEXT,
                confidence REAL,
                overall_status TEXT,
                compliance_score REAL,
                input_source TEXT,
                fields_json TEXT,
                validation_json TEXT,
                compliance_json TEXT
            )
        """)
        conn.commit()


def save_inspection(
    product_name: str,
    image_filename: Optional[str],
    engine_used: str,
    confidence: float,
    overall_status: str,
    compliance_score: float,
    input_source: str,
    fields: Dict[str, Any],
    validation_results: Dict[str, Any],
    compliance_results: Dict[str, Any],
    inspection_id: Optional[str] = None
) -> str:
    """Save or update an inspection record."""
    init_db()
    if not inspection_id:
        inspection_id = str(uuid.uuid4())[:8].upper()

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO inspections (
                id, product_name, image_filename, engine_used, confidence,
                overall_status, compliance_score, input_source,
                fields_json, validation_json, compliance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inspection_id,
            product_name or "Unknown Commodity",
            image_filename,
            engine_used,
            confidence,
            overall_status,
            compliance_score,
            input_source,
            json.dumps(fields),
            json.dumps(validation_results),
            json.dumps(compliance_results)
        ))
        conn.commit()
    return inspection_id


def get_inspection(inspection_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an inspection by its ID."""
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
        if not row:
            return None
        
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "product_name": row["product_name"],
            "image_filename": row["image_filename"],
            "engine_used": row["engine_used"],
            "confidence": row["confidence"],
            "overall_status": row["overall_status"],
            "compliance_score": row["compliance_score"],
            "input_source": row["input_source"],
            "fields": json.loads(row["fields_json"] or "{}"),
            "validation": json.loads(row["validation_json"] or "{}"),
            "compliance": json.loads(row["compliance_json"] or "{}")
        }


def list_inspections(limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """List recent inspections with optional status filter."""
    init_db()
    with get_connection() as conn:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM inspections WHERE overall_status = ? ORDER BY rowid DESC LIMIT ?",
                (status_filter, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY rowid DESC LIMIT ?",
                (limit,)
            ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "product_name": row["product_name"],
                "image_filename": row["image_filename"],
                "engine_used": row["engine_used"],
                "confidence": row["confidence"],
                "overall_status": row["overall_status"],
                "compliance_score": row["compliance_score"],
                "input_source": row["input_source"]
            })
        return results
