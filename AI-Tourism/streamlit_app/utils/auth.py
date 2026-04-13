import pandas as pd
import hashlib
import os
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.csv")

COLUMNS = ["username", "email", "password_hash", "full_name", "created_at", "last_login"]


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> pd.DataFrame:
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(USERS_FILE, index=False)
        return df
    df = pd.read_csv(USERS_FILE)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _save_users(df: pd.DataFrame):
    df.to_csv(USERS_FILE, index=False)


def user_exists(username: str) -> bool:
    df = _load_users()
    return username.strip().lower() in df["username"].str.lower().values


def email_exists(email: str) -> bool:
    df = _load_users()
    return email.strip().lower() in df["email"].str.lower().values


def register_user(username: str, email: str, password: str, full_name: str = "") -> tuple[bool, str]:
    username = username.strip()
    email = email.strip()

    if not username or not email or not password:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."
    if user_exists(username):
        return False, "Username already taken. Please choose another."
    if email_exists(email):
        return False, "An account with this email already exists."

    df = _load_users()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {
        "username": username,
        "email": email,
        "password_hash": _hash_password(password),
        "full_name": full_name.strip(),
        "created_at": now,
        "last_login": now,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_users(df)
    return True, "Account created successfully!"


def login_user(username: str, password: str) -> tuple[bool, str, dict]:
    username = username.strip()
    if not username or not password:
        return False, "Please enter your username and password.", {}

    df = _load_users()
    match = df[df["username"].str.lower() == username.lower()]
    if match.empty:
        return False, "Username not found. Please check or sign up.", {}

    row = match.iloc[0]
    if row["password_hash"] != _hash_password(password):
        return False, "Incorrect password. Please try again.", {}

    df.loc[match.index[0], "last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_users(df)

    user_info = {
        "username": row["username"],
        "email": row["email"],
        "full_name": row["full_name"] if pd.notna(row["full_name"]) else "",
        "created_at": row["created_at"],
    }
    return True, "Login successful!", user_info


def get_all_users() -> pd.DataFrame:
    return _load_users()
