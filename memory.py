"""
memory.py — Student state and progress tracking for LearnMate.

Stores and retrieves student profiles, roadmaps, completed topics,
and feedback history in a local SQLite database.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional


DB_PATH = "learnmate.db"


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they do not exist."""
    conn = _connect()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id   TEXT PRIMARY KEY,
            name         TEXT,
            domain       TEXT,
            skill_level  TEXT,
            created_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS roadmaps (
            student_id   TEXT PRIMARY KEY,
            roadmap_json TEXT,
            updated_at   TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TABLE IF NOT EXISTS completed_topics (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT,
            topic        TEXT,
            week         INTEGER,
            completed_at TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TABLE IF NOT EXISTS feedback_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT,
            feedback     TEXT,
            timestamp    TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------------
# Student CRUD
# ---------------------------------------------------------------------------

def create_student(name: str, domain: str, skill_level: str) -> str:
    """Create a new student record and return the generated student_id."""
    student_id = str(uuid.uuid4())
    conn = _connect()
    conn.execute(
        "INSERT INTO students (student_id, name, domain, skill_level, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (student_id, name, domain, skill_level, _now()),
    )
    conn.commit()
    conn.close()
    return student_id


def get_student(student_id: str) -> Optional[dict]:
    """Return student profile dict or None if not found."""
    conn = _connect()
    row = conn.execute(
        "SELECT student_id, name, domain, skill_level, created_at "
        "FROM students WHERE student_id = ?",
        (student_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "student_id": row[0],
        "name": row[1],
        "domain": row[2],
        "skill_level": row[3],
        "created_at": row[4],
    }


def update_student_level(student_id: str, skill_level: str) -> None:
    """Update the skill level for an existing student."""
    conn = _connect()
    conn.execute(
        "UPDATE students SET skill_level = ? WHERE student_id = ?",
        (skill_level, student_id),
    )
    conn.commit()
    conn.close()


def list_students() -> list[dict]:
    """Return all students (lightweight list for UI selectors)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT student_id, name, domain, skill_level FROM students ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"student_id": r[0], "name": r[1], "domain": r[2], "skill_level": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Roadmap CRUD
# ---------------------------------------------------------------------------

def save_roadmap(student_id: str, roadmap: dict) -> None:
    """Upsert the student's current roadmap."""
    conn = _connect()
    conn.execute(
        """
        INSERT INTO roadmaps (student_id, roadmap_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
            roadmap_json = excluded.roadmap_json,
            updated_at   = excluded.updated_at
        """,
        (student_id, json.dumps(roadmap), _now()),
    )
    conn.commit()
    conn.close()


def get_roadmap(student_id: str) -> Optional[dict]:
    """Return the stored roadmap dict or None."""
    conn = _connect()
    row = conn.execute(
        "SELECT roadmap_json FROM roadmaps WHERE student_id = ?",
        (student_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])


# ---------------------------------------------------------------------------
# Completed topics
# ---------------------------------------------------------------------------

def mark_topic_complete(student_id: str, topic: str, week: int) -> None:
    """Record a topic as completed for the student."""
    conn = _connect()
    # Avoid duplicates
    existing = conn.execute(
        "SELECT id FROM completed_topics WHERE student_id = ? AND topic = ?",
        (student_id, topic),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO completed_topics (student_id, topic, week, completed_at) "
            "VALUES (?, ?, ?, ?)",
            (student_id, topic, week, _now()),
        )
        conn.commit()
    conn.close()


def get_completed_topics(student_id: str) -> list[dict]:
    """Return list of completed topic records."""
    conn = _connect()
    rows = conn.execute(
        "SELECT topic, week, completed_at FROM completed_topics "
        "WHERE student_id = ? ORDER BY week, completed_at",
        (student_id,),
    ).fetchall()
    conn.close()
    return [{"topic": r[0], "week": r[1], "completed_at": r[2]} for r in rows]


def get_completed_topic_names(student_id: str) -> list[str]:
    """Return a flat list of completed topic strings."""
    return [t["topic"] for t in get_completed_topics(student_id)]


# ---------------------------------------------------------------------------
# Feedback history
# ---------------------------------------------------------------------------

def save_feedback(student_id: str, feedback: str) -> None:
    """Append a feedback entry for the student."""
    conn = _connect()
    conn.execute(
        "INSERT INTO feedback_history (student_id, feedback, timestamp) VALUES (?, ?, ?)",
        (student_id, feedback, _now()),
    )
    conn.commit()
    conn.close()


def get_feedback_history(student_id: str) -> list[dict]:
    """Return all feedback entries for the student, newest first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT feedback, timestamp FROM feedback_history "
        "WHERE student_id = ? ORDER BY timestamp DESC",
        (student_id,),
    ).fetchall()
    conn.close()
    return [{"feedback": r[0], "timestamp": r[1]} for r in rows]


def get_recent_feedback(student_id: str, n: int = 3) -> list[str]:
    """Return the n most recent feedback strings."""
    history = get_feedback_history(student_id)
    return [f["feedback"] for f in history[:n]]


# ---------------------------------------------------------------------------
# Full student state snapshot (used by the agent)
# ---------------------------------------------------------------------------

def get_student_state(student_id: str) -> Optional[dict]:
    """
    Return a consolidated state dict containing:
      - profile (name, domain, skill_level)
      - current roadmap
      - completed topics
      - recent feedback
    Returns None if the student does not exist.
    """
    student = get_student(student_id)
    if student is None:
        return None
    return {
        "profile": student,
        "roadmap": get_roadmap(student_id),
        "completed_topics": get_completed_topic_names(student_id),
        "feedback_history": get_recent_feedback(student_id, n=5),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

init_db()
