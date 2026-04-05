"""
Simple file-based authentication — stores users in users.json.
Uses bcrypt for password hashing (salted, resistant to rainbow tables).
"""

import json
import re
from pathlib import Path

import bcrypt

USERS_FILE = Path(__file__).parent / "users.json"


def _load():
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


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
    # First user gets admin role, subsequent users are regular users
    role = "admin" if len(users) == 0 else "user"
    users[email] = {"password": _hash(password), "role": role}
    _save(users)
    return True, "Account created successfully!"


def get_role(email: str) -> str:
    users = _load()
    return users.get(email, {}).get("role", "user")


def log_in(email: str, password: str) -> tuple[bool, str]:
    if not _valid_email(email):
        return False, "Invalid email address."
    users = _load()
    if email not in users:
        return False, "No account found with this email."
    if not _verify(password, users[email]["password"]):
        return False, "Incorrect password."
    return True, "Logged in successfully!"
