"""
Simple file-based authentication — stores users in users.json.
"""

import hashlib
import json
import re
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.json"


def _load():
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _valid_email(email: str) -> bool:
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


def sign_up(email: str, password: str) -> tuple[bool, str]:
    if not _valid_email(email):
        return False, "Invalid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load()
    if email in users:
        return False, "An account already exists with this email."
    users[email] = {"password": _hash(password)}
    _save(users)
    return True, "Account created successfully!"


def log_in(email: str, password: str) -> tuple[bool, str]:
    if not _valid_email(email):
        return False, "Invalid email address."
    users = _load()
    if email not in users:
        return False, "No account found with this email."
    if users[email]["password"] != _hash(password):
        return False, "Incorrect password."
    return True, "Logged in successfully!"
